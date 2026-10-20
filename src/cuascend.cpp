#include "cuascend.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
namespace ca::ir {
namespace {
bool trace() {
    static const bool on = std::getenv("CUASCEND_INTERP_DEBUG") != nullptr;
    return on;
}
constexpr int kWarpSize = 32;
constexpr size_t kLocalBytes = 64 * 1024;

// The operation of an atomic, taken from its modifiers: add, sub, min, max, and, or, xor, exch, inc, dec.
std::string atomicOperation(const ir::Instr &in) {
    static const char *const kNames[] = {"add", "sub", "min", "max",
                                         "and",  "or",  "xor", "exch",
                                         "inc",  "dec"};
    for (const char *name : kNames) {
        const std::string modifier = std::string(".") + name;
        for (const std::string &m : in.mods)
            if (m == modifier) return name;
    }
    return std::string();
}

struct Value {
    int64_t bits = 0;
    bool isFloat = false;
    uint8_t width = 32;
    bool defined = false;

    int64_t asInt() const { return bits; }

    double asDouble() const {
        if (!isFloat) return static_cast<double>(bits);
        if (width == 64) {
            double d;
            std::memcpy(&d, &bits, sizeof d);
            return d;
        }
        float f;
        const int32_t narrow = static_cast<int32_t>(bits);
        std::memcpy(&f, &narrow, sizeof f);
        return static_cast<double>(f);
    }

    static Value fromInt(int64_t v) {
        Value out;
        out.bits = v;
        return out;
    }

    static Value fromFloat(double v, uint8_t bits) {
        Value out;
        out.isFloat = true;
        out.width = bits ? bits : 32;
        if (out.width == 64) {
            std::memcpy(&out.bits, &v, sizeof v);
        } else {
            const float narrow = static_cast<float>(v);
            int32_t raw;
            std::memcpy(&raw, &narrow, sizeof raw);
            out.bits = raw;
        }
        return out;
    }
};

Value coerce(const Type &type, const Value &value) {
    if (type.isFloat())
        return Value::fromFloat(value.asDouble(), type.width ? type.width : 32);
    if (type.isInt()) {
        int64_t x = value.isFloat ? static_cast<int64_t>(value.asDouble()) : value.bits;
        switch (type.width) {
        case 8: x = static_cast<int64_t>(static_cast<int8_t>(x)); break;
        case 16: x = static_cast<int64_t>(static_cast<int16_t>(x)); break;
        case 32: x = static_cast<int64_t>(static_cast<int32_t>(x)); break;
        default: break;
        }
        return Value::fromInt(x);
    }
    return value;
}

Value decodeValue(const uint8_t *bytes, const Type &type) {
    if (type.isFloat()) {
        if (type.width == 64) {
            int64_t raw;
            std::memcpy(&raw, bytes, sizeof raw);
            Value out;
            out.isFloat = true;
            out.width = 64;
            out.bits = raw;
            return out;
        }
        int32_t raw;
        std::memcpy(&raw, bytes, sizeof raw);
        Value out;
        out.isFloat = true;
        out.width = 32;
        out.bits = raw;
        return out;
    }
    if (type.isBool()) return Value::fromInt(bytes[0] ? 1 : 0);
    switch (type.width) {
    case 8: return Value::fromInt(static_cast<int64_t>(static_cast<int8_t>(bytes[0])));
    case 16: {
        const int16_t raw = static_cast<int16_t>(bytes[0] | (bytes[1] << 8));
        return Value::fromInt(raw);
    }
    case 32: {
        const int32_t raw = static_cast<int32_t>(bytes[0] | (bytes[1] << 8) |
                                                (bytes[2] << 16) | (bytes[3] << 24));
        return Value::fromInt(raw);
    }
    default: {
        int64_t raw = 0;
        std::memcpy(&raw, bytes, sizeof raw);
        return Value::fromInt(raw);
    }
    }
}

void encodeValue(const Value &value, const Type &type, uint8_t *bytes) {
    if (type.isFloat()) {
        if (type.width == 64) {
            std::memcpy(bytes, &value.bits, sizeof value.bits);
        } else {
            const int32_t raw = static_cast<int32_t>(value.bits);
            std::memcpy(bytes, &raw, sizeof raw);
        }
        return;
    }
    if (type.isBool()) {
        bytes[0] = value.bits ? 1 : 0;
        return;
    }
    switch (type.width) {
    case 8: bytes[0] = static_cast<uint8_t>(value.bits); break;
    case 16: {
        const uint16_t raw = static_cast<uint16_t>(value.bits);
        std::memcpy(bytes, &raw, sizeof raw);
        break;
    }
    case 32: {
        const uint32_t raw = static_cast<uint32_t>(value.bits);
        std::memcpy(bytes, &raw, sizeof raw);
        break;
    }
    default: std::memcpy(bytes, &value.bits, sizeof value.bits); break;
    }
}

struct Thread {
    std::vector<Value> regs;
    size_t block = 0;   
    size_t index = 0;   
    bool active = true;
    bool waiting = false;    
    bool waitingWarp = false; 
    unsigned tx = 0, ty = 0, tz = 0;   
    unsigned cbx = 0, cby = 0, cbz = 0; 
    unsigned lane = 0;
    unsigned warp = 0;
    std::vector<uint8_t> local;
};

class Executor {
public:
    Executor(const Function &fn, const LaunchConfig &config, void **args,
             Memory &memory)
        : fn_(fn), cfg_(config), args_(args), memory_(memory) {}

    bool run(std::string *error);

private:
    bool setup(std::string *error);
    void runBlock(unsigned bx, unsigned by, unsigned bz);
    void step(Thread &thread, std::vector<Thread *> &warp);
    bool execute(const Instr &in, Thread &thread, std::vector<Thread *> &warp,
                 std::string *error);
    bool operandValue(const Thread &thread, const Operand &op, Value &out) const;
    int paramIndex(const std::string &name) const;
    bool readBytes(Space space, uint64_t address, uint8_t *dst, size_t bytes,
                   const Thread &thread) const;
    bool writeBytes(Space space, uint64_t address, const uint8_t *src,
                    size_t bytes, Thread &thread);
    void assign(Thread &thread, const Instr &in, const Value &value) const;

    const Function &fn_;
    const LaunchConfig &cfg_;
    void **args_;
    Memory &memory_;
    std::vector<Value> params_;
    std::map<std::string, int> paramByName_;
    std::vector<uint8_t> shared_;
    std::string error_;
    // The number of registers: the values are numbered densely, so the register file is a plain array.
    size_t registerCount_ = 0;
};

int Executor::paramIndex(const std::string &name) const {
    auto it = paramByName_.find(name);
    return it == paramByName_.end() ? -1 : it->second;
}

bool Executor::setup(std::string *error) {
    registerCount_ = 0;
    for (const auto &block : fn_.blocks)
        for (const auto &in : block.instrs)
            if (in.result != ir::kInvalidValue && in.result + 1 > registerCount_)
                registerCount_ = static_cast<size_t>(in.result) + 1;

    const size_t count = fn_.params.size();
    params_.assign(count, Value());
    paramByName_.clear();
    for (size_t i = 0; i < count; ++i) {
        paramByName_[fn_.params[i].name] = static_cast<int>(i);
        if (trace())
            std::fprintf(stderr, "[interp] param %zu name='%s'\n", i,
                         fn_.params[i].name.c_str());
        const Param &param = fn_.params[i];
        if (!args_ || !args_[i]) {
            *error = "kernel argument " + std::to_string(i) + " is missing";
            return false;
        }
        const bool isAddress =
            param.isPointer || (param.type.isInt() && param.type.width == 64);
        if (isAddress) {
            uint64_t address = 0;
            std::memcpy(&address, args_[i], sizeof address);
            params_[i] = Value::fromInt(static_cast<int64_t>(address));
            continue;
        }
        const size_t bytes = param.type.width / 8;
        if (bytes == 0 || bytes > 8) {
            *error = "unsupported argument type in " + param.name;
            return false;
        }
        uint8_t raw[8] = {0};
        std::memcpy(raw, args_[i], bytes);
        params_[i] = decodeValue(raw, param.type);
    }
    return true;
}

bool Executor::operandValue(const Thread &thread, const Operand &op,
                            Value &out) const {
    switch (op.kind) {
    case Operand::Kind::Value: {
        if (op.value >= thread.regs.size()) return false;
        const Value &value = thread.regs[op.value];
        if (!value.defined) return false;
        out = value;
        return true;
    }
    case Operand::Kind::Immediate:
        out = Value::fromInt(op.imm);
        return true;
    case Operand::Kind::ParamRef: {
        const int index = paramIndex(op.symbol);
        if (index < 0) return false;
        out = params_[static_cast<size_t>(index)];
        return true;
    }
    default:
        return false;
    }
}

void Executor::assign(Thread &thread, const Instr &in, const Value &value) const {
    if (in.result == kInvalidValue) return;
    if (in.result < thread.regs.size()) {
        thread.regs[in.result] = coerce(in.type, value);
        thread.regs[in.result].defined = true;
    }
}

bool Executor::readBytes(Space space, uint64_t address, uint8_t *dst,
                         size_t bytes, const Thread &thread) const {
    if (space == Space::Shared) {
        if (address + bytes > shared_.size()) return false;
        std::memcpy(dst, shared_.data() + address, bytes);
        return true;
    }
    if (space == Space::Local) {
        if (address + bytes > thread.local.size()) return false;
        std::memcpy(dst, thread.local.data() + address, bytes);
        return true;
    }
    HostView view;
    if (!memory_.view(address, view)) return false;
    const uint64_t offset = address - view.base;
    if (offset + bytes > view.size) return false;
    std::memcpy(dst, view.data + offset, bytes);
    return true;
}

bool Executor::writeBytes(Space space, uint64_t address, const uint8_t *src,
                          size_t bytes, Thread &thread) {
    if (space == Space::Shared) {
        if (address + bytes > shared_.size()) return false;
        std::memcpy(shared_.data() + address, src, bytes);
        return true;
    }
    if (space == Space::Local) {
        if (address + bytes > thread.local.size()) return false;
        std::memcpy(thread.local.data() + address, src, bytes);
        return true;
    }
    HostView view;
    if (!memory_.view(address, view)) return false;
    const uint64_t offset = address - view.base;
    if (offset + bytes > view.size) return false;
    std::memcpy(view.data + offset, src, bytes);
    return true;
}

bool Executor::run(std::string *error) {
    if (!setup(error)) return false;
    for (unsigned bz = 0; bz < cfg_.gridZ; ++bz)
        for (unsigned by = 0; by < cfg_.gridY; ++by)
            for (unsigned bx = 0; bx < cfg_.gridX; ++bx) runBlock(bx, by, bz);
    memory_.flush();
    *error = error_;
    return error_.empty();
}

void Executor::runBlock(unsigned bx, unsigned by, unsigned bz) {
    const unsigned threadsX = cfg_.blockX;
    const unsigned threadsY = cfg_.blockY;
    const unsigned threadsZ = cfg_.blockZ;
    const size_t threadCount =
        static_cast<size_t>(threadsX) * threadsY * threadsZ;
    if (threadCount == 0) return;

    std::vector<Thread> threads(threadCount);
    for (Thread &t : threads) t.regs.resize(registerCount_);
    shared_.assign(cfg_.sharedBytes, 0);

    unsigned index = 0;
    for (unsigned tz = 0; tz < threadsZ; ++tz)
        for (unsigned ty = 0; ty < threadsY; ++ty)
            for (unsigned tx = 0; tx < threadsX; ++tx, ++index) {
                Thread &t = threads[index];
                t.cbx = bx;
                t.cby = by;
                t.cbz = bz;
                t.tx = tx;
                t.ty = ty;
                t.tz = tz;
                t.lane = index % kWarpSize;
                t.warp = index / kWarpSize;
                t.local.assign(kLocalBytes, 0);
            }

    const unsigned warps = static_cast<unsigned>((threadCount + kWarpSize - 1) / kWarpSize);

    for (;;) {
        bool executed = false;
        for (unsigned w = 0; w < warps; ++w) {
            std::vector<Thread *> warp;
            const size_t first = static_cast<size_t>(w) * kWarpSize;
            for (size_t l = 0; l < kWarpSize && first + l < threadCount; ++l)
                warp.push_back(&threads[first + l]);
            for (Thread *thread : warp) {
                if (!thread->active || thread->waiting || thread->waitingWarp)
                    continue;
                executed = true;
                step(*thread, warp);
                if (!error_.empty()) return;
            }
        }

        
        
        for (unsigned w = 0; w < warps; ++w) {
            const size_t first = static_cast<size_t>(w) * kWarpSize;
            bool allWaiting = true;
            bool anyActive = false;
            for (size_t l = 0; l < kWarpSize && first + l < threadCount; ++l) {
                const Thread &t = threads[first + l];
                if (!t.active) continue;
                anyActive = true;
                if (!t.waitingWarp) allWaiting = false;
            }
            if (anyActive && allWaiting)
                for (size_t l = 0; l < kWarpSize && first + l < threadCount; ++l)
                    threads[first + l].waitingWarp = false;
        }

        bool anyActive = false;
        bool anyRunning = false;
        for (const Thread &t : threads) {
            if (!t.active) continue;
            anyActive = true;
            if (!t.waiting && !t.waitingWarp) anyRunning = true;
        }
        if (anyRunning) continue;
        if (anyActive) {
            for (Thread &t : threads) {
                t.waiting = false;
                t.waitingWarp = false;
            }
            continue;
        }
        if (!executed) break;
    }
}

void Executor::step(Thread &thread, std::vector<Thread *> &warp) {
    if (thread.block >= fn_.blocks.size()) {
        thread.active = false;
        return;
    }
    const Block &block = fn_.blocks[thread.block];
    if (thread.index >= block.instrs.size()) {
        
        
        thread.block += 1;
        thread.index = 0;
        if (thread.block >= fn_.blocks.size()) thread.active = false;
        return;
    }
    const Instr &in = block.instrs[thread.index];
    if (trace()) {
        std::fprintf(stderr, "[interp] block %zu instr %zu: %s%s\n", thread.block,
                     thread.index, ir::opName(in.op), in.origText.c_str());
    }

    
    
    if (in.hasPred && in.op != Op::CondBr && in.op != Op::Br) {
        Value predicate;
        if (!operandValue(thread, Operand::val(in.pred), predicate)) {
            error_ = "predicate is not defined";
            thread.active = false;
            return;
        }
        if (predicate.asInt() == 0) {
            thread.index++;
            return;
        }
    }

    std::string error;
    if (!execute(in, thread, warp, &error)) {
        error_ = error.empty() ? std::string("instruction is not modelled: ") +
                                      ir::opName(in.op)
                                : error;
        if (in.line) error_ += " (PTX line " + std::to_string(in.line) + ")";
        if (!in.origText.empty()) error_ += " [" + in.origText + "]";
        thread.active = false;
        return;
    }
}

bool Executor::execute(const Instr &in, Thread &thread,
                       std::vector<Thread *> &warp, std::string *error) {
    auto next = [&]() { thread.index++; };

    switch (in.op) {
    case Op::Ret:
        thread.active = false;
        return true;

    case Op::Br:
        if (in.args.empty() || in.args[0].kind != Operand::Kind::LabelRef)
            return false;
        thread.block = in.args[0].block;
        thread.index = 0;
        return true;

    case Op::CondBr: {
        if (in.args.empty() || in.args[0].kind != Operand::Kind::LabelRef)
            return false;
        
        
        
        bool taken = true;
        if (in.hasPred && in.pred != ir::kInvalidValue) {
            Value predicate;
            if (!operandValue(thread, Operand::val(in.pred), predicate)) return false;
            taken = predicate.asInt() != 0;
        }
        thread.block = taken ? in.args[0].block : thread.block + 1;
        thread.index = 0;
        return true;
    }

    case Op::Mov: {
        
        
        
        if (in.args.empty()) return false;
        Value value;
        if (!operandValue(thread, in.args[0], value)) return false;
        if (in.result != ir::kInvalidValue) {
            if (in.type.isFloat()) {
                value.isFloat = true;
                value.width = in.type.width ? in.type.width : 32;
            } else if (in.type.isInt()) {
                value.isFloat = false;
            }
            if (in.result < thread.regs.size()) {
                thread.regs[in.result] = value;
                thread.regs[in.result].defined = true;
            }
        }
        next();
        return true;
    }

    case Op::Cvt: {
        if (in.args.empty()) return false;
        Value value;
        if (!operandValue(thread, in.args[0], value)) return false;
        assign(thread, in, value);
        next();
        return true;
    }

    case Op::Selp: {
        if (in.args.size() < 3) return false;
        Value condition, a, b;
        if (!operandValue(thread, in.args[0], condition) ||
            !operandValue(thread, in.args[1], a) ||
            !operandValue(thread, in.args[2], b))
            return false;
        assign(thread, in, condition.asInt() ? a : b);
        next();
        return true;
    }

    case Op::Load: {
        
        
        
        if (in.addr.base.kind == Operand::Kind::ParamRef) {
            Value value;
            if (!operandValue(thread, in.addr.base, value)) return false;
            assign(thread, in, value);
            next();
            return true;
        }
        uint64_t address = 0;
        Value base;
        if (!operandValue(thread, in.addr.base, base)) {
            *error = "the address of the access is not a value";
            return false;
        }
        int64_t effective = base.asInt();
        if (in.addr.hasIndex) {
            Value index;
            if (!operandValue(thread, in.addr.index, index)) return false;
            effective += index.asInt() * in.addr.scale;
        }
        effective += in.addr.offset;
        address = static_cast<uint64_t>(effective);
        const size_t width = in.type.width ? in.type.width / 8 : 4;
        uint8_t raw[8] = {0};
        if (!readBytes(in.space, address, raw, width, thread)) {
            *error = "read of " + std::to_string(width) + " bytes at " +
                     std::to_string(address) + " is outside a known buffer";
            return false;
        }
        assign(thread, in, decodeValue(raw, in.type));
        next();
        return true;
    }

    case Op::Store: {
        if (in.args.empty()) return false;
        Value base, value;
        if (!operandValue(thread, in.addr.base, base)) {
            *error = "the address of the access is not a value";
            return false;
        }
        if (!operandValue(thread, in.args[0], value)) return false;
        int64_t effective = base.asInt();
        if (in.addr.hasIndex) {
            Value index;
            if (!operandValue(thread, in.addr.index, index)) return false;
            effective += index.asInt() * in.addr.scale;
        }
        effective += in.addr.offset;
        const Type type = in.type.isFloat() || in.type.isInt() ? in.type : Type{TypeKind::Float, 32, VecLen::Scalar};
        const size_t width = type.width ? type.width / 8 : 4;
        uint8_t raw[8] = {0};
        encodeValue(coerce(type, value), type, raw);
        if (!writeBytes(in.space, static_cast<uint64_t>(effective), raw, width, thread)) {
            *error = "write outside of a known buffer";
            return false;
        }
        next();
        return true;
    }

    
    case Op::AtomicRMW:
    case Op::AtomicCAS: {
        // An atomic reads the old value, computes the new one and writes it back; the threads run one after another, so the order differs from the hardware, but the result is the same on every run.
        Value base;
        if (!operandValue(thread, in.addr.base, base)) {
            *error = "the address of the atomic is not a value";
            return false;
        }
        int64_t effective = base.asInt();
        if (in.addr.hasIndex) {
            Value index;
            if (!operandValue(thread, in.addr.index, index)) return false;
            effective += index.asInt() * in.addr.scale;
        }
        effective += in.addr.offset;
        const Type type = in.type.isFloat() || in.type.isInt()
                              ? in.type
                              : Type{TypeKind::Int, 32, VecLen::Scalar};
        const size_t width = type.width ? type.width / 8 : 4;
        if (width == 0 || width > 8) return false;

        uint8_t raw[8] = {0};
        if (!readBytes(in.space, static_cast<uint64_t>(effective), raw, width,
                       thread)) {
            *error = "the atomic touches memory outside a known buffer";
            return false;
        }
        const Value old = decodeValue(raw, type);

        Value operand;
        if (!operandValue(thread, in.args[0], operand)) return false;
        Value replacement = operand;
        if (in.op == Op::AtomicCAS && in.args.size() > 1) {
            Value other;
            if (!operandValue(thread, in.args[1], other)) return false;
            replacement = other;
        }

        Value result = old;
        if (in.op == Op::AtomicCAS) {
            if (old.asInt() == operand.asInt()) result = replacement;
        } else {
            const std::string operation = atomicOperation(in);
            if (operation == "add")
                result = type.isFloat()
                             ? Value::fromFloat(old.asDouble() + operand.asDouble(),
                                                type.width)
                             : Value::fromInt(old.asInt() + operand.asInt());
            else if (operation == "sub")
                result = type.isFloat()
                             ? Value::fromFloat(old.asDouble() - operand.asDouble(),
                                                type.width)
                             : Value::fromInt(old.asInt() - operand.asInt());
            else if (operation == "min")
                result = type.isFloat()
                             ? Value::fromFloat(std::min(old.asDouble(),
                                                        operand.asDouble()),
                                                type.width)
                             : Value::fromInt(std::min(old.asInt(), operand.asInt()));
            else if (operation == "max")
                result = type.isFloat()
                             ? Value::fromFloat(std::max(old.asDouble(),
                                                        operand.asDouble()),
                                                type.width)
                             : Value::fromInt(std::max(old.asInt(), operand.asInt()));
            else if (operation == "and")
                result = Value::fromInt(old.asInt() & operand.asInt());
            else if (operation == "or")
                result = Value::fromInt(old.asInt() | operand.asInt());
            else if (operation == "xor")
                result = Value::fromInt(old.asInt() ^ operand.asInt());
            else if (operation == "exch")
                result = operand;
            else if (operation == "inc")
                result = Value::fromInt(old.asInt() + 1);
            else if (operation == "dec")
                result = Value::fromInt(old.asInt() - 1);
            else {
                *error = "the atomic operation '" + operation + "' is not modelled";
                return false;
            }
        }

        encodeValue(result, type, raw);
        if (!writeBytes(in.space, static_cast<uint64_t>(effective), raw, width,
                        thread)) {
            *error = "the atomic writes outside a known buffer";
            return false;
        }
        if (in.result != ir::kInvalidValue && in.result < thread.regs.size()) {
            thread.regs[in.result] = coerce(type, result);
            thread.regs[in.result].defined = true;
        }
        next();
        return true;
    }

    case Op::Add: case Op::Sub: case Op::Mul: case Op::DivS: case Op::DivU:
    case Op::RemS: case Op::RemU: case Op::Neg: case Op::Not: case Op::Shl:
    case Op::ShrS: case Op::ShrU: case Op::And: case Op::Or: case Op::Xor:
    case Op::MinS: case Op::MinU: case Op::MaxS: case Op::MaxU: {
        if (in.args.size() < 1) return false;
        Value a;
        if (!operandValue(thread, in.args[0], a)) return false;
        Value b;
        const bool binary = in.op != Op::Neg && in.op != Op::Not;
        if (binary && (!operandValue(thread, in.args[1], b))) return false;
        int64_t x = a.asInt();
        int64_t y = binary ? b.asInt() : 0;
        int64_t result = 0;
        switch (in.op) {
        case Op::Add: result = x + y; break;
        case Op::Sub: result = x - y; break;
        case Op::Mul: result = x * y; break;
        case Op::DivS: if (y == 0) return false; result = x / y; break;
        case Op::DivU: if (y == 0) return false;
            result = static_cast<int64_t>(static_cast<uint64_t>(x) / static_cast<uint64_t>(y));
            break;
        case Op::RemS: if (y == 0) return false; result = x % y; break;
        case Op::RemU: if (y == 0) return false;
            result = static_cast<int64_t>(static_cast<uint64_t>(x) % static_cast<uint64_t>(y));
            break;
        case Op::Neg: result = -x; break;
        case Op::Not: result = ~x; break;
        case Op::Shl: result = x << (y & 63); break;
        case Op::ShrS: result = x >> (y & 63); break;
        case Op::ShrU: result = static_cast<int64_t>(static_cast<uint64_t>(x) >> (y & 63)); break;
        case Op::And: result = x & y; break;
        case Op::Or: result = x | y; break;
        case Op::Xor: result = x ^ y; break;
        case Op::MinS: result = x < y ? x : y; break;
        case Op::MinU: result = static_cast<uint64_t>(x) < static_cast<uint64_t>(y) ? x : y; break;
        case Op::MaxS: result = x > y ? x : y; break;
        case Op::MaxU: result = static_cast<uint64_t>(x) > static_cast<uint64_t>(y) ? x : y; break;
        default: break;
        }
        assign(thread, in, Value::fromInt(result));
        next();
        return true;
    }

    
    case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv: case Op::FRem:
    case Op::FFma: case Op::FMulAdd: case Op::FNeg: case Op::FAbs:
    case Op::Exp: case Op::Exp2: case Op::Log: case Op::Log2: case Op::Log10:
    case Op::Sqrt: case Op::RSqrt: case Op::Sin: case Op::Cos: case Op::Tan:
    case Op::Pow: case Op::Floor: case Op::Ceil: case Op::Round: case Op::Trunc:
    case Op::Fma3: case Op::MinF: case Op::MaxF: {
        if (in.args.empty()) return false;
        Value a;
        if (!operandValue(thread, in.args[0], a)) return false;
        const double x = a.asDouble();
        double y = 0;
        const bool binary = !(in.op == Op::FNeg || in.op == Op::FAbs ||
                              in.op == Op::Exp || in.op == Op::Exp2 ||
                              in.op == Op::Log || in.op == Op::Log2 ||
                              in.op == Op::Log10 || in.op == Op::Sqrt ||
                              in.op == Op::RSqrt || in.op == Op::Sin ||
                              in.op == Op::Cos || in.op == Op::Tan ||
                              in.op == Op::Floor || in.op == Op::Ceil ||
                              in.op == Op::Round || in.op == Op::Trunc);
        if (binary) {
            Value b;
            if (!operandValue(thread, in.args[1], b)) return false;
            y = b.asDouble();
        }
        double result = 0;
        switch (in.op) {
        case Op::FAdd: result = x + y; break;
        case Op::FSub: result = x - y; break;
        case Op::FMul: result = x * y; break;
        case Op::FDiv: if (y == 0) return false; result = x / y; break;
        case Op::FRem: if (y == 0) return false; result = std::fmod(x, y); break;
        case Op::FFma: case Op::Fma3: {
            if (in.args.size() < 3) return false;
            Value c;
            if (!operandValue(thread, in.args[2], c)) return false;
            result = std::fma(x, y, c.asDouble());
            break;
        }
        case Op::FMulAdd: {
            if (in.args.size() < 3) return false;
            Value c;
            if (!operandValue(thread, in.args[2], c)) return false;
            result = x * y + c.asDouble();
            break;
        }
        case Op::FNeg: result = -x; break;
        case Op::FAbs: result = std::fabs(x); break;
        case Op::Exp: result = std::exp(x); break;
        case Op::Exp2: result = std::exp2(x); break;
        case Op::Log: result = std::log(x); break;
        case Op::Log2: result = std::log2(x); break;
        case Op::Log10: result = std::log10(x); break;
        case Op::Sqrt: result = std::sqrt(x); break;
        case Op::RSqrt: result = 1.0 / std::sqrt(x); break;
        case Op::Sin: result = std::sin(x); break;
        case Op::Cos: result = std::cos(x); break;
        case Op::Tan: result = std::tan(x); break;
        case Op::Pow: result = std::pow(x, y); break;
        case Op::Floor: result = std::floor(x); break;
        case Op::Ceil: result = std::ceil(x); break;
        case Op::Round: result = std::round(x); break;
        case Op::Trunc: result = std::trunc(x); break;
        case Op::MinF: result = x < y ? x : y; break;
        case Op::MaxF: result = x > y ? x : y; break;
        default: break;
        }
        assign(thread, in, Value::fromFloat(result, in.type.width ? in.type.width : 32));
        next();
        return true;
    }

    
    case Op::CmpEq: case Op::CmpNe: case Op::CmpLtS: case Op::CmpLeS:
    case Op::CmpGtS: case Op::CmpGeS: case Op::CmpLtU: case Op::CmpLeU:
    case Op::CmpGtU: case Op::CmpGeU:
    case Op::FCmpEq: case Op::FCmpNe: case Op::FCmpLt: case Op::FCmpLe:
    case Op::FCmpGt: case Op::FCmpGe: {
        if (in.args.size() < 2) return false;
        Value a, b;
        if (!operandValue(thread, in.args[0], a) ||
            !operandValue(thread, in.args[1], b))
            return false;
        const bool isFloat = in.op == Op::FCmpEq || in.op == Op::FCmpNe ||
                             in.op == Op::FCmpLt || in.op == Op::FCmpLe ||
                             in.op == Op::FCmpGt || in.op == Op::FCmpGe;
        bool result = false;
        if (isFloat) {
            const double x = a.asDouble();
            const double y = b.asDouble();
            switch (in.op) {
            case Op::FCmpEq: result = x == y; break;
            case Op::FCmpNe: result = x != y; break;
            case Op::FCmpLt: result = x < y; break;
            case Op::FCmpLe: result = x <= y; break;
            case Op::FCmpGt: result = x > y; break;
            default: result = x >= y; break;
            }
        } else {
            const int64_t x = a.asInt();
            const int64_t y = b.asInt();
            switch (in.op) {
            case Op::CmpEq: result = x == y; break;
            case Op::CmpNe: result = x != y; break;
            case Op::CmpLtS: result = x < y; break;
            case Op::CmpLeS: result = x <= y; break;
            case Op::CmpGtS: result = x > y; break;
            case Op::CmpGeS: result = x >= y; break;
            case Op::CmpLtU:
                result = static_cast<uint64_t>(x) < static_cast<uint64_t>(y); break;
            case Op::CmpLeU:
                result = static_cast<uint64_t>(x) <= static_cast<uint64_t>(y); break;
            case Op::CmpGtU:
                result = static_cast<uint64_t>(x) > static_cast<uint64_t>(y); break;
            default:
                result = static_cast<uint64_t>(x) >= static_cast<uint64_t>(y); break;
            }
        }
        assign(thread, in, Value::fromInt(result ? 1 : 0));
        next();
        return true;
    }

    
    case Op::TidX: assign(thread, in, Value::fromInt(thread.tx)); next(); return true;
    case Op::TidY: assign(thread, in, Value::fromInt(thread.ty)); next(); return true;
    case Op::TidZ: assign(thread, in, Value::fromInt(thread.tz)); next(); return true;
    case Op::CtaidX: assign(thread, in, Value::fromInt(thread.cbx)); next(); return true;
    case Op::CtaidY: assign(thread, in, Value::fromInt(thread.cby)); next(); return true;
    case Op::CtaidZ: assign(thread, in, Value::fromInt(thread.cbz)); next(); return true;
    case Op::LaneId: assign(thread, in, Value::fromInt(thread.lane)); next(); return true;
    case Op::WarpId: assign(thread, in, Value::fromInt(thread.warp)); next(); return true;
    case Op::NtidX: assign(thread, in, Value::fromInt(cfg_.blockX)); next(); return true;
    case Op::NtidY: assign(thread, in, Value::fromInt(cfg_.blockY)); next(); return true;
    case Op::NtidZ: assign(thread, in, Value::fromInt(cfg_.blockZ)); next(); return true;

    
    
    
    
    case Op::BarrierThreads:
        thread.waiting = true;
        next();
        return true;
    case Op::BarrierWarp:
        thread.waitingWarp = true;
        next();
        return true;

    
    case Op::ShflIdx: case Op::ShflUp: case Op::ShflDown: case Op::ShflXor:
    case Op::ShflPred: {
        
        
        if (in.args.size() < 2) return false;
        const size_t sourceArg = in.op == Op::ShflPred ? 1 : 0;
        const size_t offsetArg = in.op == Op::ShflPred ? 2 : 1;
        Value source, offset;
        if (!operandValue(thread, in.args[sourceArg], source)) return false;
        if (!operandValue(thread, in.args[offsetArg], offset)) return false;
        const int lane = thread.lane;
        int from = lane;
        switch (in.op) {
        case Op::ShflIdx:
            from = static_cast<int>(offset.asInt()) & (kWarpSize - 1);
            break;
        case Op::ShflUp: {
            const int delta = static_cast<int>(offset.asInt());
            from = lane - delta;
            if (from < 0) from = lane;
            break;
        }
        case Op::ShflDown: {
            const int delta = static_cast<int>(offset.asInt());
            from = lane + delta;
            if (from >= kWarpSize) from = lane;
            break;
        }
        case Op::ShflXor:
            from = lane ^ (static_cast<int>(offset.asInt()) & (kWarpSize - 1));
            break;
        default: {
            
            
            bool found = false;
            for (Thread *peer : warp) {
                if (!peer->active) continue;
                Value condition;
                if (!operandValue(*peer, in.args[0], condition)) return false;
                if (condition.asInt() == 0) continue;
                Value candidate;
                if (!operandValue(*peer, in.args[sourceArg], candidate)) return false;
                assign(thread, in, candidate);
                found = true;
                break;
            }
            if (!found) assign(thread, in, Value::fromInt(0));
            next();
            return true;
        }
        }
        if (from < 0 || static_cast<size_t>(from) >= warp.size()) return false;
        Thread *peer = warp[static_cast<size_t>(from)];
        Value candidate;
        if (!operandValue(*peer, in.args[sourceArg], candidate)) return false;
        assign(thread, in, candidate);
        next();
        return true;
    }

    case Op::VoteAll: case Op::VoteAny: case Op::VoteBallot: case Op::Ballot:
    case Op::ActiveMask: {
        
        
        uint32_t mask = 0;
        int activeLanes = 0;
        for (size_t l = 0; l < warp.size(); ++l)
            if (warp[l]->active) {
                mask |= 1u << l;
                ++activeLanes;
            }
        switch (in.op) {
        case Op::VoteAll: assign(thread, in, Value::fromInt(activeLanes ? 1 : 0)); break;
        case Op::VoteAny: assign(thread, in, Value::fromInt(activeLanes ? 1 : 0)); break;
        case Op::VoteBallot:
        case Op::Ballot:
        case Op::ActiveMask:
        default: assign(thread, in, Value::fromInt(mask)); break;
        }
        next();
        return true;
    }

    case Op::Popc: {
        if (in.args.empty()) return false;
        Value value;
        if (!operandValue(thread, in.args[0], value)) return false;
        assign(thread, in, Value::fromInt(__builtin_popcountll(
            static_cast<uint64_t>(value.asInt()))));
        next();
        return true;
    }

    case Op::NumWarps:
        assign(thread, in, Value::fromInt(
            (cfg_.blockX * cfg_.blockY * cfg_.blockZ + kWarpSize - 1) / kWarpSize));
        next();
        return true;

    default:
        return false;
    }
}

} 

bool interpret(const Function &fn, const LaunchConfig &config, void **args,
               Memory &memory, std::string *error) {
    std::string local;
    Executor executor(fn, config, args, memory);
    const bool ok = executor.run(&local);
    if (error) *error = local;
    return ok;
}

} 

// MAIN: the PTX parser

#include <cctype>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <string_view>

namespace ca::ptx {
namespace {

bool isIdentStart(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}
bool isIdentChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}

} 

std::vector<Token> lex(const std::string &text) {
    std::vector<Token> out;
    size_t i = 0;
    int line = 1, col = 1;
    const size_t n = text.size();

    auto advance = [&](size_t count) {
        for (size_t k = 0; k < count; ++k) {
            if (text[i] == '\n') {
                ++line;
                col = 1;
            } else {
                ++col;
            }
            ++i;
        }
    };

    auto push = [&](Token::Kind k, std::string t, int l, int c) {
        out.push_back(Token{k, std::move(t), l, c});
    };

    while (i < n) {
        const char c = text[i];

        
        if (std::isspace(static_cast<unsigned char>(c)) || c == '\0') {
            advance(1);
            continue;
        }

        
        if (c == '/' && i + 1 < n && text[i + 1] == '/') {
            while (i < n && text[i] != '\n') advance(1);
            continue;
        }
        if (c == '/' && i + 1 < n && text[i + 1] == '*') {
            advance(2);
            while (i + 1 < n && !(text[i] == '*' && text[i + 1] == '/')) advance(1);
            if (i + 1 < n) advance(2);
            continue;
        }

        const int startLine = line, startCol = col;

        
        if (c == '.') {
            size_t j = i + 1;
            while (j < n && isIdentChar(text[j])) ++j;
            
            push(Token::Kind::Directive, text.substr(i, j - i), startLine, startCol);
            advance(j - i);
            continue;
        }

        
        if (c == '%') {
            size_t j = i + (i + 1 < n && text[i + 1] == '%' ? 2 : 1);
            while (j < n && (isIdentChar(text[j]) || text[j] == '.')) ++j;
            if (j == i + 1 || (i + 1 < n && text[i + 1] == '%' && j == i + 2))
                throw Error("bare '%' in PTX", startLine, startCol);
            push(Token::Kind::Register, text.substr(i, j - i), startLine, startCol);
            advance(j - i);
            continue;
        }

        
        if (c == '$') {
            size_t j = i + 1;
            while (j < n && isIdentChar(text[j])) ++j;
            push(Token::Kind::Param, text.substr(i, j - i), startLine, startCol);
            advance(j - i);
            continue;
        }

        
        if (c == '"') {
            size_t j = i + 1;
            while (j < n && text[j] != '"') {
                if (text[j] == '\\' && j + 1 < n) ++j;
                ++j;
            }
            if (j >= n) throw Error("unterminated string", startLine, startCol);
            push(Token::Kind::String, text.substr(i + 1, j - i - 1), startLine,
                 startCol);
            advance(j - i + 1);
            continue;
        }

        
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            ((c == '-' || c == '+') && i + 1 < n &&
             (std::isdigit(static_cast<unsigned char>(text[i + 1])) ||
              text[i + 1] == '.'))) {
            size_t j = i;
            if (text[j] == '-' || text[j] == '+') ++j;

            const bool hexInt =
                j + 1 < n && text[j] == '0' && (text[j + 1] == 'x' || text[j + 1] == 'X');
            const bool hexFloat =
                j + 1 < n && text[j] == '0' && (text[j + 1] == 'f' || text[j + 1] == 'F');

            if (hexInt || hexFloat) {
                j += 2;
                while (j < n && std::isxdigit(static_cast<unsigned char>(text[j]))) ++j;
            } else {
                bool isDecimalFloat = false;
                while (j < n && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
                if (j < n && text[j] == '.') {
                    isDecimalFloat = true;
                    ++j;
                    while (j < n && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
                }
                
                if (j < n && (text[j] == 'e' || text[j] == 'E')) {
                    size_t k = j + 1;
                    if (k < n && (text[k] == '+' || text[k] == '-')) ++k;
                    if (k < n && std::isdigit(static_cast<unsigned char>(text[k]))) {
                        j = k;
                        while (j < n && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
                        isDecimalFloat = true;
                    }
                }
                
                if (j < n && std::strchr("fFlLuU", text[j])) ++j;
                const bool plainFloat = isDecimalFloat && j < n &&
                                        std::strchr("fF", text[j - 1]) != nullptr;
                push((isDecimalFloat || plainFloat) ? Token::Kind::Float
                                                    : Token::Kind::Number,
                     text.substr(i, j - i), startLine, startCol);
                advance(j - i);
                continue;
            }

            push(hexFloat ? Token::Kind::Float : Token::Kind::Number,
                 text.substr(i, j - i), startLine, startCol);
            advance(j - i);
            continue;
        }

        
        if (isIdentStart(c)) {
            size_t j = i;
            while (j < n && isIdentChar(text[j])) ++j;
            push(Token::Kind::Ident, text.substr(i, j - i), startLine, startCol);
            advance(j - i);
            continue;
        }

        
        if (std::strchr("()[]{},;:@!<>=|&+-*", c)) {
            push(Token::Kind::Punct, std::string(1, c), startLine, startCol);
            advance(1);
            continue;
        }

        throw Error(std::string("unexpected character '") + c + "'", startLine,
                    startCol);
    }

    push(Token::Kind::End, "", line, col);
    return out;
}

namespace {

class Parser {
public:
    Parser(std::vector<Token> toks, std::string filename)
        : t_(std::move(toks)), file_(std::move(filename)) {}

    Module parseModule();

private:
    std::vector<Token> t_;
    std::string file_;
    size_t p_ = 0;

    const Token &cur() const { return t_[p_]; }
    const Token &peek(size_t n = 1) const {
        return t_[p_ + n < t_.size() ? p_ + n : t_.size() - 1];
    }
    bool atEnd() const { return cur().kind == Token::Kind::End; }
    void bump() {
        if (!atEnd()) ++p_;
    }

    [[noreturn]] void fail(const std::string &msg) const {
        throw Error(file_ + ": " + msg, cur().line, cur().col);
    }

    
    int64_t parseNumberToken();

    bool isDirective(const char *name) const {
        return cur().kind == Token::Kind::Directive &&
               cur().text == name;
    }
    void expectDirective(const char *name) {
        if (!isDirective(name))
            fail(std::string("expected '") + name + "', got '" + cur().text + "'");
        bump();
    }
    void expectPunct(const char *p) {
        if (cur().kind != Token::Kind::Punct || cur().text != p)
            fail(std::string("expected '") + p + "', got '" + cur().text + "'");
        bump();
    }

    
    Type parseType();
    StateSpace parseSpace();
    Type parseParamType();

    
    Operand parseOperand();
    Operand parseAddrOperand();
    Operand parseVectorOperand();
    Operand parseOperandPrimary();

    
    Inst parseInst();

    
    Function parseFunction(bool visible);
    void parseBody(std::vector<Inst> &out, std::vector<RegDecl> &fnRegs);

    
    std::vector<std::string> knownTypes_;
    void noteType(const Type &t) {
        const std::string s = t.str();
        for (const auto &k : knownTypes_)
            if (k == s) return;
        knownTypes_.push_back(s);
    }
    bool isKnownType(const std::string &s) const {
        for (const auto &k : knownTypes_)
            if (k == s) return true;
        return false;
    }
};

Type Parser::parseType() {
    Type t;
    if (cur().kind != Token::Kind::Directive)
        fail("expected a type directive, got '" + cur().text + "'");

    const std::string d = cur().text;
    
    if (d == ".v2" || d == ".v4") {
        t.vecLen = static_cast<unsigned>(d[2] - '0');
        bump();
    }
    if (cur().kind != Token::Kind::Directive)
        fail("expected scalar type after vector prefix");

    const std::string s = cur().text;
    
    auto widthOf = [](const char *digits) {
        unsigned w = 0;
        for (; *digits && *digits >= '0' && *digits <= '9'; ++digits)
            w = w * 10 + static_cast<unsigned>(*digits - '0');
        return w;
    };
    if (s == ".b8" || s == ".b16" || s == ".b32" || s == ".b64") {
        t.kind = TypeKind::Bool;
        t.width = widthOf(s.c_str() + 2);
    } else if (s == ".pred") {
        t.kind = TypeKind::Bool;
        t.width = 1;
    } else if (s[0] == '.' && s.size() >= 4 && (s[1] == 's' || s[1] == 'u')) {
        t.kind = TypeKind::Int;
        t.width = widthOf(s.c_str() + 2);
    } else if (s[0] == '.' && s.size() >= 4 && s[1] == 'f') {
        t.kind = TypeKind::Float;
        t.width = widthOf(s.c_str() + 2);
    } else {
        fail("unknown type directive '" + s + "'");
    }

    
    if (t.kind == TypeKind::Float && t.width == 16 && t.vecLen == 1) {
        
    }
    bump();
    return t;
}

StateSpace Parser::parseSpace() {
    if (cur().kind != Token::Kind::Directive) return StateSpace::Generic;
    if (isDirective(".global")) {
        bump();
        return StateSpace::Global;
    }
    if (isDirective(".shared")) {
        bump();
        return StateSpace::Shared;
    }
    if (isDirective(".local")) {
        bump();
        return StateSpace::Local;
    }
    if (isDirective(".const")) {
        bump();
        return StateSpace::Constant;
    }
    if (isDirective(".param")) {
        bump();
        return StateSpace::Param;
    }
    return StateSpace::Generic;
}

Type Parser::parseParamType() {
    
    Type t;
    t.space = StateSpace::Param;
    bool isPtr = false;

    for (;;) {
        if (cur().kind != Token::Kind::Directive) break;

        if (isDirective(".ptr")) {
            bump();
            isPtr = true;
            continue;
        }

        const StateSpace sp = parseSpace();
        if (sp != StateSpace::Generic) {
            t.space = sp;
            continue;
        }

        if (isDirective(".align")) {
            bump();
            if (cur().kind == Token::Kind::Number ||
                cur().kind == Token::Kind::Float)
                bump();
            continue;
        }

        
        break;
    }

    t = parseType();
    if (t.space == StateSpace::Generic) t.space = StateSpace::Param;

    
    while (isDirective(".align")) {
        bump();
        if (cur().kind == Token::Kind::Number || cur().kind == Token::Kind::Float)
            bump();
    }

    (void)isPtr;
    return t;
}

Operand Parser::parseAddrOperand() {
    
    Operand o;
    o.kind = OperandKind::Addr;

    for (;;) {
        
        std::string reg;
        bool isParam = false;
        bool haveTerm = false;

        if (cur().kind == Token::Kind::Register) {
            reg = cur().text.substr(1);
            bump();
            haveTerm = true;
        } else if (cur().kind == Token::Kind::Param) {
            reg = cur().text;
            bump();
            haveTerm = true;
            isParam = true;
        } else if (cur().kind == Token::Kind::Ident) {
            
            reg = cur().text;
            bump();
            haveTerm = true;
            isParam = true;
        }

        if (haveTerm) {
            if (!o.addr.hasBase) {
                o.addr.baseReg = reg;
                o.addr.hasBase = true;
                o.addr.isParamRef = isParam;
            } else if (!o.addr.hasIndex) {
                o.addr.indexReg = reg;
                o.addr.hasIndex = true;
            } else {
                fail("too many registers in address expression");
            }
        } else if (cur().kind == Token::Kind::Number ||
                   cur().kind == Token::Kind::Float) {
            o.addr.offset += parseNumberToken();
            bump();
        } else if (cur().kind == Token::Kind::Punct && cur().text == "-") {
            bump();
            if (cur().kind != Token::Kind::Number &&
                cur().kind != Token::Kind::Float)
                fail("expected number after '-' in address");
            o.addr.offset -= parseNumberToken();
            bump();
        } else {
            break;
        }

        
        if (cur().kind == Token::Kind::Number ||
            cur().kind == Token::Kind::Float) {
            o.addr.offset += parseNumberToken();
            bump();
            continue;
        }

        if (cur().kind != Token::Kind::Punct) break;

        
        if (cur().text == ",") {
            bump();
            continue;
        }

        if (cur().text == "*") {
            
            bump();
            if (cur().kind != Token::Kind::Number &&
                cur().kind != Token::Kind::Float)
                fail("expected scale after '*' in address");
            if (!o.addr.hasIndex)
                fail("scale '*' without an index register in address");
            o.addr.scale = parseNumberToken();
            bump();
            continue;
        }

        if (cur().text == "+" || cur().text == "-") {
            const bool neg = cur().text == "-";
            bump();
            
            if (cur().kind == Token::Kind::Number ||
                cur().kind == Token::Kind::Float) {
                const int64_t v = parseNumberToken();
                bump();
                if (neg) o.addr.offset -= v;
                else o.addr.offset += v;
            }
            continue;
        }

        break;
    }

    expectPunct("]");
    return o;
}

Operand Parser::parseVectorOperand() {
    Operand o;
    o.kind = OperandKind::Vector;
    expectPunct("{");
    while (!(cur().kind == Token::Kind::Punct && cur().text == "}")) {
        if (atEnd()) fail("unterminated vector literal");
        o.elements.push_back(parseOperandPrimary());
        if (cur().kind == Token::Kind::Punct && cur().text == ",") bump();
    }
    expectPunct("}");
    return o;
}

Operand Parser::parseOperandPrimary() {
    Operand o;

    switch (cur().kind) {
    case Token::Kind::Register: {
        const std::string t = cur().text;
        
        o.kind = (t.size() >= 2 && t[1] == '%') ? OperandKind::Label
                                               : OperandKind::Register;
        o.name = (o.kind == OperandKind::Label) ? t.substr(2) : t.substr(1);
        bump();
        return o;
    }
    case Token::Kind::Param:
        o.kind = OperandKind::Param;
        o.name = cur().text.substr(1);
        bump();
        return o;
    case Token::Kind::Number:
    case Token::Kind::Float:
        o.kind = OperandKind::Immediate;
        o.immText = cur().text;
        o.imm = parseNumberToken();
        bump();
        return o;
    case Token::Kind::String:
        o.kind = OperandKind::Symbol;
        o.name = cur().text;
        bump();
        return o;
    case Token::Kind::Ident:
        
        o.kind = OperandKind::Label;
        o.name = cur().text;
        bump();
        return o;
    case Token::Kind::Directive: {
        
        const StateSpace sp = parseSpace();
        if (sp != StateSpace::Generic) {
            o.kind = OperandKind::Constant;
            o.space = sp;
            return o;
        }
        fail("unexpected directive '" + cur().text + "' in operand position");
    }
    case Token::Kind::Punct:
        if (cur().text == "[") {
            bump();
            return parseAddrOperand();
        }
        if (cur().text == "(") {
            
            bump();
            Operand o;
            o.kind = OperandKind::Vector;
            while (!(cur().kind == Token::Kind::Punct && cur().text == ")")) {
                if (atEnd()) fail("unterminated call result list");
                o.elements.push_back(parseOperandPrimary());
                if (cur().kind == Token::Kind::Punct && cur().text == ",") bump();
            }
            expectPunct(")");
            return o;
        }
        if (cur().text == "{") {
            return parseVectorOperand();
        }
        if (cur().text == "<") { 
            bump();
            o.kind = OperandKind::TypeRef;
            while (!(cur().kind == Token::Kind::Punct && cur().text == ">")) {
                if (atEnd()) fail("unterminated <type>");
                bump();
            }
            expectPunct(">");
            return o;
        }
        fail("unexpected '" + cur().text + "' in operand position");
    default:
        fail("unexpected token '" + cur().text + "'");
    }
}

Operand Parser::parseOperand() {
    
    if (cur().kind == Token::Kind::Register) {
        Operand first = parseOperandPrimary();
        if (first.kind == OperandKind::Register &&
            cur().kind == Token::Kind::Register) {
            const bool nextIsDelim =
                (peek().kind == Token::Kind::Punct &&
                 (peek().text == "," || peek().text == "[" ||
                  peek().text == ";")) ||
                (peek().kind == Token::Kind::End);
            if (nextIsDelim) {
                Operand vec;
                vec.kind = OperandKind::Vector;
                vec.elements.push_back(first);
                vec.elements.push_back(parseOperandPrimary());
                return vec;
            }
        }
        return first;
    }
    return parseOperandPrimary();
}

Inst Parser::parseInst() {
    Inst in;
    in.line = cur().line;
    in.col = cur().col;

    
    if (cur().kind == Token::Kind::Punct && cur().text == "@") {
        bump();
        in.hasPred = true;
        if (cur().kind == Token::Kind::Punct && cur().text == "!") {
            in.pred = "!";
            bump();
        }
        if (cur().kind == Token::Kind::Register) {
            in.pred += cur().text.substr(1);
            bump();
        } else if (cur().kind == Token::Kind::Ident) {
            in.pred += cur().text;
            bump();
        } else {
            fail("expected predicate register after '@'");
        }
    }

    
    if (cur().kind == Token::Kind::Ident) {
        in.opcode = cur().text;
        bump();

        
        static const char *const kNameOps[] = {
            
            "bra", "call", "ret", "brx", "br", "exit", "trap", "setp",
            
            "shfl", "vote", "ballot", "atom", "red", "bar", "membar",
            "griddepcontrol", "prefetch", "remap", "wait", "cp", "elect"};
        static const char *const kStructural[] = {
            "sync", "down", "up", "idx", "bfly", "clamp", "warp", "all",
            "any", "ballot", "uni", "popc", "noinc", "aligned", "shared",
            "cta", "gl", "sys"};

        auto isNameOp = [&](const std::string &op) {
            for (const char *c : kNameOps)
                if (op == c) return true;
            return false;
        };
        auto isStructural = [&](const std::string &m) {
            
            if (m.empty() || m[0] != '.') return false;
            for (const char *c : kStructural)
                if (m.c_str() + 1 == std::string_view(c)) return true;
            return false;
        };

        if (isNameOp(in.opcode)) {
            while (cur().kind == Token::Kind::Directive &&
                   isStructural(cur().text)) {
                in.opcode += cur().text;
                bump();
            }
        }
    } else if (cur().kind == Token::Kind::Directive) {
        
        fail("expected opcode, got directive '" + cur().text + "'");
    } else {
        fail("expected instruction opcode, got '" + cur().text + "'");
    }

    
    std::vector<std::string> mods;
    while (cur().kind == Token::Kind::Directive) {
        mods.push_back(cur().text);
        bump();
    }
    in.modifiers = mods;

    
    if (cur().kind == Token::Kind::Punct && cur().text == ";") {
        bump();
        return in;
    }

    for (;;) {
        in.operands.push_back(parseOperand());
        if (cur().kind == Token::Kind::Punct && cur().text == ",") {
            bump();
            continue;
        }
        break;
    }

    if (!(cur().kind == Token::Kind::Punct && cur().text == ";"))
        fail("expected ';' at end of instruction '" + in.opcode + "'");
    bump();
    return in;
}

void Parser::parseBody(std::vector<Inst> &out, std::vector<RegDecl> &fnRegs) {
    while (!atEnd()) {
        
        if (cur().kind == Token::Kind::Punct && cur().text == "}") break;

        
        if ((cur().kind == Token::Kind::Ident ||
             cur().kind == Token::Kind::Register) &&
            peek().kind == Token::Kind::Punct && peek().text == ":") {
            Inst lab;
            lab.isLabel = true;
            lab.line = cur().line;
            lab.label = cur().text;
            if (!lab.label.empty() && lab.label[0] == '%') lab.label.erase(0, 1);
            bump();
            bump();
            out.push_back(std::move(lab));
            continue;
        }
        if (cur().kind == Token::Kind::Directive && cur().text == ".loc") {
            
            while (!(cur().kind == Token::Kind::Punct && cur().text == ";")) {
                if (atEnd()) fail("unterminated .loc");
                bump();
            }
            bump();
            continue;
        }
        
        if (cur().kind == Token::Kind::Directive) {
            const std::string d = cur().text;
            RegKind rk;
            if (d == ".reg") rk = RegKind::Reg;
            else if (d == ".local") rk = RegKind::Local;
            else if (d == ".shared") rk = RegKind::Shared;
            else if (d == ".global") rk = RegKind::Global;
            else if (d == ".const") rk = RegKind::Const;
            else rk = RegKind::Reg;

            if (d == ".reg" || d == ".local" || d == ".shared" ||
                d == ".global" || d == ".const") {
                bump(); 

                RegDecl rd;
                rd.isLocal = (rk == RegKind::Local);
                switch (rk) {
                case RegKind::Local: rd.space = StateSpace::Local; break;
                case RegKind::Shared: rd.space = StateSpace::Shared; break;
                case RegKind::Global: rd.space = StateSpace::Global; break;
                case RegKind::Const: rd.space = StateSpace::Constant; break;
                default: rd.space = StateSpace::Generic; break;
                }

                
                for (;;) {
                    rd.type = parseType();
                    noteType(rd.type);

                    
                    if (cur().kind == Token::Kind::Register) {
                        rd.name = cur().text.substr(1);
                        bump();
                    } else if (cur().kind == Token::Kind::Param) {
                        rd.name = cur().text.substr(1);
                        bump();
                    } else {
                        fail("expected register name in " + d + " declaration");
                    }

                    
                    if (cur().kind == Token::Kind::Punct && cur().text == "<") {
                        bump();
                        unsigned n = 0;
                        if (cur().kind == Token::Kind::Number) {
                            n = static_cast<unsigned>(parseNumberToken());
                            bump();
                        }
                        expectPunct(">");
                        rd.dims.push_back(n);
                    } else if (cur().kind == Token::Kind::Punct &&
                               cur().text == "[") {
                        bump();
                        unsigned n = 0;
                        if (cur().kind == Token::Kind::Number) {
                            n = static_cast<unsigned>(parseNumberToken());
                            bump();
                        }
                        expectPunct("]");
                        rd.dims.push_back(n);
                    }

                    fnRegs.push_back(rd);

                    if (cur().kind == Token::Kind::Punct && cur().text == ",") {
                        bump();
                        
                        continue;
                    }
                    break;
                }

                if (!(cur().kind == Token::Kind::Punct && cur().text == ";"))
                    fail("expected ';' after " + d + " declaration");
                bump();
                continue;
            }

            if (d == ".pragma" || d == ".section" || d == ".file") {
                while (!(cur().kind == Token::Kind::Punct && cur().text == ";")) {
                    if (atEnd()) fail("unterminated " + d);
                    bump();
                }
                bump();
                continue;
            }
        }
        out.push_back(parseInst());
    }
}

Function Parser::parseFunction(bool visible) {
    Function f;
    f.visible = visible;

    
    if (cur().kind != Token::Kind::Ident && cur().kind != Token::Kind::Register)
        fail("expected kernel/function name");
    f.name = cur().text;
    bump();

    expectPunct("(");
    if (!(cur().kind == Token::Kind::Punct && cur().text == ")")) {
        for (;;) {
            Param pm;
            pm.type = parseParamType();
            
            pm.isPointer = pm.type.kind == TypeKind::Int && pm.type.width == 64;

            
            if (cur().kind == Token::Kind::Param) {
                pm.name = cur().text;
                bump();
            } else if (cur().kind == Token::Kind::Register) {
                pm.name = cur().text;
                bump();
            } else if (cur().kind == Token::Kind::Ident) {
                pm.name = cur().text;
                bump();
            }

            
            while (cur().kind == Token::Kind::Punct && cur().text == "[") {
                bump();
                unsigned d = 0;
                if (cur().kind == Token::Kind::Number) {
                    d = static_cast<unsigned>(parseNumberToken());
                    bump();
                }
                pm.arrayDims.push_back(d);
                expectPunct("]");
            }

            noteType(pm.type);
            f.params.push_back(pm);

            if (cur().kind == Token::Kind::Punct && cur().text == ",") {
                bump();
                continue;
            }
            break;
        }
    }
    expectPunct(")");

    
    for (;;) {
        
        if (isDirective(".maxntid") || isDirective(".minntid") ||
            isDirective(".maxnreg") || isDirective(".reqntid") ||
            isDirective(".launch_bounds")) {
            const bool isMaxNtid = isDirective(".maxntid");
            const bool isMinNtid = isDirective(".minntid");
            bump();

            uint32_t vals[3] = {0, 0, 0};
            unsigned nvals = 0;
            for (;;) {
                if (cur().kind == Token::Kind::Number ||
                    cur().kind == Token::Kind::Float) {
                    if (nvals < 3) vals[nvals] = static_cast<uint32_t>(parseNumberToken());
                    ++nvals;
                    bump();
                }
                if (cur().kind == Token::Kind::Punct && cur().text == ",") {
                    bump();
                    continue;
                }
                break;
            }
            if (isMaxNtid && nvals > 0) f.maxThreads = vals[0];
            if (isMinNtid && nvals > 0) f.minThreads = vals[0];
            if (isMaxNtid && nvals > 2) f.maxBlocks = vals[2];

            
            if (cur().kind == Token::Kind::Punct && cur().text == ",") {
                bump();
                if (cur().kind == Token::Kind::Number ||
                    cur().kind == Token::Kind::Float)
                    bump();
            }
            continue;
        }
        if (isDirective(".section") || isDirective(".align") ||
            isDirective(".loc") || isDirective(".pragma")) {
            while (!(cur().kind == Token::Kind::Punct && cur().text == ";")) {
                if (atEnd()) fail("unterminated " + cur().text);
                bump();
            }
            bump();
            continue;
        }
        break;
    }

    
    if (cur().kind == Token::Kind::Punct && cur().text == ";") {
        bump();
        f.external = true;
        return f;
    }

    expectPunct("{");
    parseBody(f.body, f.regs);
    expectPunct("}");
    return f;
}

int64_t Parser::parseNumberToken() {
    const std::string &t = cur().text;

    
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'f' || t[1] == 'F')) {
        return static_cast<int64_t>(std::strtoull(t.c_str() + 2, nullptr, 16));
    }
    
    return static_cast<int64_t>(std::strtoll(t.c_str(), nullptr, 0));
}

Module Parser::parseModule() {
    Module m;

    while (!atEnd()) {
        if (cur().kind != Token::Kind::Directive) {
            fail("expected a module-level directive, got '" + cur().text + "'");
        }

        if (isDirective(".version")) {
            bump();
            
            if (cur().kind == Token::Kind::Number ||
                cur().kind == Token::Kind::Float) {
                const std::string t = cur().text;
                const size_t dot = t.find('.');
                if (dot != std::string::npos) {
                    m.versionMajor = std::atoi(t.substr(0, dot).c_str());
                    m.versionMinor = std::atoi(t.substr(dot + 1).c_str());
                } else {
                    m.versionMajor = std::atoi(t.c_str());
                }
                bump();
            }
            continue;
        }
        if (isDirective(".target")) {
            bump();
            if (cur().kind == Token::Kind::Ident ||
                cur().kind == Token::Kind::Directive) {
                m.target = cur().text;
                bump();
            }
            continue;
        }
        if (isDirective(".address_size")) {
            bump();
            if (cur().kind == Token::Kind::Number) {
                m.addressSize = std::atoi(cur().text.c_str());
                bump();
            }
            continue;
        }

        
        if (isDirective(".visible") || isDirective(".weak") ||
            isDirective(".extern")) {
            bump();
            continue; 
        }

        if (isDirective(".entry")) {
            bump();
            m.functions.push_back(parseFunction(true));
            continue;
        }

        if (isDirective(".func")) {
            bump();
            m.functions.push_back(parseFunction(false));
            continue;
        }

        
        if (isDirective(".extern") || isDirective(".visible")) continue;

        
        bump();
        while (cur().kind != Token::Kind::Punct && cur().text != ";" &&
               !atEnd() && cur().kind == Token::Kind::Directive) {
            bump();
        }
        if (cur().kind == Token::Kind::Punct && cur().text == ";") bump();
    }

    return m;
}

} 

Module parse(const std::string &text, const std::string &filename) {
    Parser parser(lex(text), filename);
    return parser.parseModule();
}

const Function *Module::findEntry(const std::string &name) const {
    for (const auto &f : functions)
        if (f.name == name) return &f;
    return nullptr;
}

} 

//  MAIN: the fatbin reader

#include <elf.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

namespace ca::fatbin {
namespace {

constexpr uint32_t kFatbinMagic = 0x466243b1;

constexpr size_t kHeaderSize = 16;
constexpr size_t kEntrySize = 64;

const char *const kFatbinSections[] = {
    ".nv_fatbin",
    ".nvFatBin",
    "__nv_relfatbin",
};

class Blob {
public:
    Blob(const uint8_t *p, size_t n) : p_(p), n_(n) {}

    void require(size_t off, size_t len, const char *what) const {
        if (off > n_ || len > n_ - off)
            throw Error(std::string("out of bounds reading ") + what);
    }

    uint8_t u8(size_t off) const {
        require(off, 1, "u8");
        return p_[off];
    }
    uint16_t u16(size_t off) const {
        require(off, 2, "u16");
        uint16_t v;
        std::memcpy(&v, p_ + off, 2);
        return v;
    }
    uint32_t u32(size_t off) const {
        require(off, 4, "u32");
        uint32_t v;
        std::memcpy(&v, p_ + off, 4);
        return v;
    }
    uint64_t u64(size_t off) const {
        require(off, 8, "u64");
        uint64_t v;
        std::memcpy(&v, p_ + off, 8);
        return v;
    }

    const uint8_t *raw(size_t off, size_t len) const {
        require(off, len, "slice");
        return p_ + off;
    }
    size_t size() const { return n_; }

private:
    const uint8_t *p_;
    size_t n_;
};

struct ShdrView {
    uint32_t name;
    uint64_t offset;
    uint64_t size;
};

ShdrView readShdr(const Blob &elf, size_t at, bool is64) {
    ShdrView v{};
    v.name = elf.u32(at + 0x00);
    v.offset = is64 ? elf.u64(at + 0x18) : elf.u32(at + 0x10);
    v.size = is64 ? elf.u64(at + 0x20) : elf.u32(at + 0x14);
    return v;
}

bool findFatbinSection(const Blob &elf, size_t &outOff, size_t &outSize) {
    if (elf.size() < sizeof(Elf64_Ehdr)) return false;

    const bool is64 = elf.u8(4) == ELFCLASS64;
    const uint16_t shoff = is64 ? elf.u16(0x28) : elf.u16(0x20);
    const uint16_t shentsize = is64 ? elf.u16(0x3A) : elf.u16(0x2E);
    const uint16_t shnum = is64 ? elf.u16(0x3C) : elf.u16(0x30);
    const uint16_t shstrndx = is64 ? elf.u16(0x3E) : elf.u16(0x32);

    if (shoff == 0 || shnum == 0 || shstrndx >= shnum) return false;
    if (static_cast<size_t>(shoff) + static_cast<size_t>(shnum) * shentsize >
        elf.size())
        return false;

    
    const size_t strHdr = shoff + static_cast<size_t>(shstrndx) * shentsize;
    const ShdrView strtab = readShdr(elf, strHdr, is64);
    const uint8_t *strBase = elf.raw(static_cast<size_t>(strtab.offset),
                                     static_cast<size_t>(strtab.size));

    for (uint16_t i = 0; i < shnum; ++i) {
        const size_t sh = shoff + static_cast<size_t>(i) * shentsize;
        const ShdrView s = readShdr(elf, sh, is64);
        if (s.name == 0 || s.size == 0) continue;
        if (s.name >= strtab.size) continue;

        
        const uint8_t *name = strBase + s.name;
        const size_t avail = static_cast<size_t>(strtab.size) - s.name;
        const size_t maxCmp = avail < 64 ? avail : 64;

        for (const char *want : kFatbinSections) {
            const size_t wlen = std::strlen(want);
            if (wlen <= maxCmp && std::memcmp(name, want, wlen) == 0 &&
                name[wlen] == 0) {
                elf.require(static_cast<size_t>(s.offset),
                            static_cast<size_t>(s.size), "fatbin section");
                outOff = static_cast<size_t>(s.offset);
                outSize = static_cast<size_t>(s.size);
                return true;
            }
        }
    }
    return false;
}

void parseFatbin(const uint8_t *data, size_t size, Fatbin &out, int depth) {
    
    if (depth > 4) throw Error("fatbin nesting too deep");

    const Blob fb(data, size);
    if (fb.size() < kHeaderSize)
        throw Error("fatbin shorter than header");

    const uint32_t magic = fb.u32(0);
    if (magic != kFatbinMagic) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "bad fatbin magic 0x%08x", magic);
        throw Error(buf);
    }

    const uint16_t headerSize = fb.u16(6);
    const uint64_t fatSize = fb.u64(8);
    if (headerSize < kHeaderSize)
        throw Error("fatbin headerSize smaller than struct");

    
    size_t limit = fb.size();
    if (fatSize >= headerSize && fatSize <= fb.size())
        limit = static_cast<size_t>(fatSize);

    size_t pos = headerSize;
    while (pos + kEntrySize <= limit) {
        const uint32_t kindRaw = fb.u32(pos + 0);
        const uint64_t payloadSize = fb.u64(pos + 16);

        Entry e;
        switch (kindRaw) {
        case 1: e.kind = Kind::PTX; break;
        case 2: e.kind = Kind::Cubin; break;
        case 4: e.kind = Kind::Compressed; break;
        default: e.kind = Kind::Unknown; break;
        }

        
        if (e.kind == Kind::Cubin) {
            e.sm_major = fb.u32(pos + 28);
            e.sm_minor = fb.u32(pos + 32);
        }

        const size_t dataOff = pos + headerSize;
        fb.require(dataOff, static_cast<size_t>(payloadSize), "entry payload");

        if (e.kind == Kind::Compressed) {
            
            std::vector<uint8_t> inner = lz4::decodeBlock(
                fb.raw(dataOff, static_cast<size_t>(payloadSize)),
                static_cast<size_t>(payloadSize));
            
            const uint32_t idx = out.intern(std::move(inner));
            parseFatbin(out.bufferAt(idx), out.bufferSize(idx), out, depth + 1);
        } else if (e.kind != Kind::Unknown) {
            const uint8_t *payload = fb.raw(dataOff, 0) + dataOff;
            
            if (depth == 0) {
                e.buffer = 0;
                e.offset = dataOff;
                e.size = payloadSize;
                out.addInputEntry(e);
            } else {
                std::vector<uint8_t> copy(payload,
                                          payload + static_cast<size_t>(payloadSize));
                e.buffer = out.intern(std::move(copy));
                e.offset = 0;
                e.size = payloadSize;
                out.addOwnedEntry(e);
            }
        }

        
        pos += headerSize;
        pos += (payloadSize + 15) & ~static_cast<size_t>(15);
    }
}

bool startsWithPtx(const Blob &b) {
    
    size_t i = 0;
    while (i < b.size()) {
        const uint8_t c = b.u8(i);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
            c == '\v') {
            ++i;
            continue;
        }
        break;
    }
    if (i >= b.size()) return false;

    const uint8_t c = b.u8(i);
    if (c == '#') return true; 

    
    if (c == '/' && i + 1 < b.size()) {
        const uint8_t c2 = b.u8(i + 1);
        if (c2 == '/' || c2 == '*') return true;
    }

    
    if (c != '.') return false;
    static const char *const kDirectives[] = {".version", ".target",
                                              ".address_size", ".entry",
                                              ".func", ".visible"};
    for (const char *d : kDirectives) {
        const size_t n = std::strlen(d);
        if (i + n <= b.size() &&
            std::memcmp(b.raw(i, n), d, n) == 0)
            return true;
    }
    return false;
}

} 

uint32_t Fatbin::intern(std::vector<uint8_t> buf) {
    owned_.push_back(std::move(buf));
    return static_cast<uint32_t>(owned_.size()); 
}

void Fatbin::addInputEntry(Entry e) {
    entries_.push_back(e);
}

void Fatbin::addOwnedEntry(Entry e) {
    entries_.push_back(e);
}

void Fatbin::setInput(std::vector<uint8_t> bytes) {
    input_ = std::move(bytes);
}

const uint8_t *Fatbin::bufferAt(uint32_t index) const {
    if (index == 0) return input_.data();
    if (index > owned_.size()) throw Error("entry references unknown buffer");
    return owned_[index - 1].data();
}

size_t Fatbin::bufferSize(uint32_t index) const {
    if (index == 0) return input_.size();
    if (index > owned_.size()) throw Error("entry references unknown buffer");
    return owned_[index - 1].size();
}

const uint8_t *Fatbin::data(const Entry &e) const {
    const size_t bsz = bufferSize(e.buffer);
    if (e.offset > bsz || e.size > bsz - e.offset)
        throw Error("entry payload outside buffer");
    return bufferAt(e.buffer) + e.offset;
}

std::string Fatbin::text(const Entry &e) const {
    const uint8_t *p = data(e);
    return std::string(reinterpret_cast<const char *>(p),
                       static_cast<size_t>(e.size));
}

std::string Fatbin::firstPtx() const {
    for (const auto &e : entries_)
        if (e.kind == Kind::PTX) return text(e);
    return {};
}

bool Fatbin::hasPtx() const {
    for (const auto &e : entries_)
        if (e.kind == Kind::PTX) return true;
    return false;
}

bool Fatbin::hasCubin() const {
    for (const auto &e : entries_)
        if (e.kind == Kind::Cubin) return true;
    return false;
}

Format detect(const void *data, size_t size) {
    const Blob b(static_cast<const uint8_t *>(data), size);
    if (size >= 4 && std::memcmp(b.raw(0, 4), "\x7f" "ELF", 4) == 0) {
        size_t off = 0, sz = 0;
        if (findFatbinSection(b, off, sz)) return Format::FatbinELF;
        return Format::CubinELF;
    }
    if (startsWithPtx(b)) return Format::RawPTX;
    return Format::Unknown;
}

Fatbin parse(const void *data, size_t size) {
    Fatbin out;
    const Blob b(static_cast<const uint8_t *>(data), size);

    
    const uint8_t *src = b.raw(0, size);
    out.setInput(std::vector<uint8_t>(src, src + size));

    switch (detect(data, size)) {
    case Format::FatbinELF: {
        size_t off = 0, sz = 0;
        if (!findFatbinSection(b, off, sz))
            throw Error("ELF present but no fatbin section found");
        const uint8_t *inp = out.bufferAt(0);
        parseFatbin(inp + off, sz, out, 0);
        return out;
    }
    case Format::CubinELF: {
        
        Entry e;
        e.kind = Kind::Cubin;
        e.buffer = 0;
        e.offset = 0;
        e.size = size;
        out.addInputEntry(e);
        return out;
    }
    case Format::RawPTX: {
        Entry e;
        e.kind = Kind::PTX;
        e.buffer = 0;
        e.offset = 0;
        e.size = size;
        out.addInputEntry(e);
        return out;
    }
    case Format::Unknown:
        throw Error("input is neither ELF/fatbin, cubin, nor PTX text");
    }
    throw Error("unreachable");
}

Fatbin parseFile(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw Error("cannot open " + path);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    if (buf.empty()) throw Error("empty file: " + path);
    return parse(buf.data(), buf.size());
}

std::string archName(uint32_t sm_major, uint32_t sm_minor) {
    return "sm_" + std::to_string(sm_major) + std::to_string(sm_minor);
}

} 

//  MAIN: the PTX printer

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace ca::ptx {
namespace {

const char *spaceName(StateSpace s) {
    switch (s) {
    case StateSpace::Global: return ".global";
    case StateSpace::Shared: return ".shared";
    case StateSpace::Local: return ".local";
    case StateSpace::Constant: return ".const";
    case StateSpace::Param: return ".param";
    case StateSpace::Generic: break;
    }
    return "";
}

} 

unsigned RegDecl::slots() const {
    
    if (!dims.empty()) return dims[0];
    
    size_t i = 0;
    while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i]))) ++i;
    if (i == 0) return 1;
    const int n = std::atoi(name.c_str());
    return n > 0 ? static_cast<unsigned>(n) : 1;
}

const Type *Function::registerType(const std::string &regName) const {
    if (regName.empty()) return nullptr;

    
    size_t n = 0;
    while (n < regName.size() &&
           !std::isdigit(static_cast<unsigned char>(regName[n])))
        ++n;
    if (n == 0) return nullptr;
    const std::string prefix = regName.substr(0, n);

    
    for (auto it = regs.rbegin(); it != regs.rend(); ++it) {
        if (it->name == prefix) return &it->type;
    }
    return nullptr;
}

std::string Type::str() const {
    std::ostringstream os;
    if (vecLen > 1) os << ".v" << vecLen;
    switch (kind) {
    case TypeKind::Void: return "void";
    case TypeKind::Bool:
        if (width == 1) os << ".pred";
        else os << ".b" << width;
        break;
    case TypeKind::Int:
        os << (isSigned() ? ".s" : ".u") << width;
        break;
    case TypeKind::Float:
        os << ".f" << width;
        break;
    }
    if (space != StateSpace::Generic) os << " " << spaceName(space);
    return os.str();
}

std::string Operand::str() const {
    std::ostringstream os;
    switch (kind) {
    case OperandKind::Immediate:
        os << immText;
        break;
    case OperandKind::Register:
        os << "%" << name;
        break;
    case OperandKind::Label:
        os << "%%" << name;
        break;
    case OperandKind::Param:
        os << "$" << name;
        break;
    case OperandKind::Local:
    case OperandKind::Global:
    case OperandKind::Constant:
        os << spaceName(space);
        break;
    case OperandKind::Special:
        os << "%" << name;
        break;
    case OperandKind::Symbol:
    case OperandKind::FuncRef:
        os << name;
        break;
    case OperandKind::TypeRef:
        os << "<type>";
        break;
    case OperandKind::Addr:
        os << "[";
        if (addr.hasBase) os << "%" << addr.baseReg;
        if (addr.hasIndex) os << " + %" << addr.indexReg;
        if (addr.offset != 0) {
            if (addr.hasBase || addr.hasIndex) os << " + ";
            os << addr.offset;
        }
        os << "]";
        break;
    case OperandKind::Vector:
        os << "{";
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i) os << ", ";
            os << elements[i].str();
        }
        os << "}";
        break;
    }
    return os.str();
}

bool Inst::hasMod(const std::string &m) const {
    for (const auto &s : modifiers)
        if (s == m) return true;
    return false;
}

std::string Inst::modsJoined() const {
    std::string out;
    for (const auto &m : modifiers) {
        out += m;
        out += " ";
    }
    return out;
}

std::string dump(const Module &m) {
    std::ostringstream os;
    os << "// PTX version " << m.versionMajor << "." << m.versionMinor
       << "  target " << m.target << "  addr_size " << m.addressSize << "\n";

    for (const auto &f : m.functions) {
        os << (f.visible ? "\n.visible .entry " : "\n.func ") << f.name << "(";
        for (size_t i = 0; i < f.params.size(); ++i) {
            if (i) os << ", ";
            const auto &p = f.params[i];
            os << p.type.str();
            if (p.type.width) os << " ";
            os << p.name;
            for (auto d : p.arrayDims) os << "[" << d << "]";
        }
        os << ")\n";
        if (f.maxThreads)
            os << "// .maxntid " << f.maxThreads << " " << f.minThreads << " "
               << f.maxBlocks << "\n";
        if (f.external) {
            os << "  ; // external\n";
            continue;
        }
        os << "{\n";
        for (const auto &in : f.body) {
            if (in.isLabel) {
                os << in.label << ":\n";
                continue;
            }
            os << "  " << in.line << ":" << in.col << "  ";
            if (in.hasPred) os << "@" << in.pred << " ";
            os << in.opcode;
            for (const auto &mod : in.modifiers) os << mod;
            for (const auto &op : in.operands) os << " " << op.str();
            os << ";\n";
        }
        os << "}\n";
    }
    return os.str();
}

} 

// MAIN: the IR

#include <sstream>

namespace ca::ir {

const char *spaceName(Space s) {
    switch (s) {
    case Space::Global: return "global";
    case Space::Shared: return "shared";
    case Space::Local: return "local";
    case Space::Constant: return "const";
    case Space::Param: return "param";
    case Space::Generic: break;
    }
    return "generic";
}

const char *opName(Op o) {
    switch (o) {
    case Op::Ret: return "ret";
    case Op::Br: return "br";
    case Op::CondBr: return "cond_br";
    case Op::Call: return "call";
    case Op::Unreachable: return "unreachable";

    case Op::Add: return "add";
    case Op::Sub: return "sub";
    case Op::Mul: return "mul";
    case Op::DivS: return "div.s";
    case Op::DivU: return "div.u";
    case Op::RemS: return "rem.s";
    case Op::RemU: return "rem.u";
    case Op::Neg: return "neg";
    case Op::Shl: return "shl";
    case Op::ShrS: return "shr.s";
    case Op::ShrU: return "shr.u";
    case Op::And: return "and";
    case Op::Or: return "or";
    case Op::Xor: return "xor";
    case Op::Not: return "not";
    case Op::MinS: return "min.s";
    case Op::MinU: return "min.u";
    case Op::MaxS: return "max.s";
    case Op::MaxU: return "max.u";
    case Op::CmpEq: return "cmp.eq";
    case Op::CmpNe: return "cmp.ne";
    case Op::CmpLtS: return "cmp.lt.s";
    case Op::CmpLeS: return "cmp.le.s";
    case Op::CmpGtS: return "cmp.gt.s";
    case Op::CmpGeS: return "cmp.ge.s";
    case Op::CmpLtU: return "cmp.lt.u";
    case Op::CmpLeU: return "cmp.le.u";
    case Op::CmpGtU: return "cmp.gt.u";
    case Op::CmpGeU: return "cmp.ge.u";

    case Op::FAdd: return "fadd";
    case Op::FSub: return "fsub";
    case Op::FMul: return "fmul";
    case Op::FDiv: return "fdiv";
    case Op::FRem: return "frem";
    case Op::FFma: return "ffma";
    case Op::FNeg: return "fneg";
    case Op::FAbs: return "fabs";
    case Op::FMulAdd: return "fmuladd";
    case Op::FCmpEq: return "fcmp.eq";
    case Op::FCmpNe: return "fcmp.ne";
    case Op::FCmpLt: return "fcmp.lt";
    case Op::FCmpLe: return "fcmp.le";
    case Op::FCmpGt: return "fcmp.gt";
    case Op::FCmpGe: return "fcmp.ge";
    case Op::MinF: return "min.f";
    case Op::MaxF: return "max.f";

    case Op::Exp: return "exp";
    case Op::Exp2: return "exp2";
    case Op::Log: return "log";
    case Op::Log2: return "log2";
    case Op::Log10: return "log10";
    case Op::Sqrt: return "sqrt";
    case Op::RSqrt: return "rsqrt";
    case Op::Sin: return "sin";
    case Op::Cos: return "cos";
    case Op::Tan: return "tan";
    case Op::SinCos: return "sincos";
    case Op::Pow: return "pow";
    case Op::Floor: return "floor";
    case Op::Ceil: return "ceil";
    case Op::Round: return "round";
    case Op::Trunc: return "trunc";
    case Op::Fma3: return "fma3";

    case Op::Cvt: return "cvt";
    case Op::Selp: return "selp";
    case Op::Mov: return "mov";
    case Op::Set: return "set";

    case Op::Load: return "load";
    case Op::Store: return "store";

    case Op::BarrierThreads: return "bar.sync";
    case Op::BarrierWarp: return "bar.warp.sync";

    case Op::ShflIdx: return "shfl.idx";
    case Op::ShflUp: return "shfl.up";
    case Op::ShflDown: return "shfl.down";
    case Op::ShflXor: return "shfl.xor";
    case Op::ShflPred: return "shfl.pred";
    case Op::VoteAll: return "vote.all";
    case Op::VoteAny: return "vote.any";
    case Op::VoteBallot: return "vote.ballot";
    case Op::Ballot: return "ballot";
    case Op::Popc: return "popc";
    case Op::ActiveMask: return "activemask";
    case Op::LaneId: return "laneid";
    case Op::WarpId: return "warpid";
    case Op::NumWarps: return "numwarps";

    case Op::TidX: return "%tid.x";
    case Op::TidY: return "%tid.y";
    case Op::TidZ: return "%tid.z";
    case Op::CtaidX: return "%ctaid.x";
    case Op::CtaidY: return "%ctaid.y";
    case Op::CtaidZ: return "%ctaid.z";
    case Op::NtidX: return "%ntid.x";
    case Op::NtidY: return "%ntid.y";
    case Op::NtidZ: return "%ntid.z";

    case Op::AtomicRMW: return "atomic.rmw";
    case Op::AtomicCAS: return "atomic.cas";
    case Op::Unknown: return "unknown";
    }
    return "?";
}

bool isWarpOp(Op o) {
    switch (o) {
    case Op::ShflIdx:
    case Op::ShflUp:
    case Op::ShflDown:
    case Op::ShflXor:
    case Op::ShflPred:
    case Op::VoteAll:
    case Op::VoteAny:
    case Op::VoteBallot:
    case Op::Ballot:
    case Op::Popc:
    case Op::ActiveMask:
    case Op::LaneId:
    case Op::WarpId:
    case Op::NumWarps:
    case Op::BarrierWarp:
        return true;
    default:
        return false;
    }
}

bool isTerminator(Op o) {
    return o == Op::Ret || o == Op::Br || o == Op::CondBr ||
           o == Op::Call || o == Op::Unreachable;
}

bool isFloatOp(Op o) {
    switch (o) {
    case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv:
    case Op::FRem: case Op::FFma: case Op::FNeg: case Op::FAbs:
    case Op::FMulAdd:
    case Op::FCmpEq: case Op::FCmpNe: case Op::FCmpLt:
    case Op::FCmpLe: case Op::FCmpGt: case Op::FCmpGe:
    case Op::Exp: case Op::Exp2: case Op::Log: case Op::Log2:
    case Op::Log10: case Op::Sqrt: case Op::RSqrt: case Op::Sin:
    case Op::Cos: case Op::Tan: case Op::SinCos: case Op::Pow:
    case Op::Floor: case Op::Ceil: case Op::Round: case Op::Trunc:
    case Op::Fma3: case Op::MinF: case Op::MaxF:
        return true;
    default:
        return false;
    }
}

bool isComparison(Op o) {
    switch (o) {
    case Op::CmpEq: case Op::CmpNe:
    case Op::CmpLtS: case Op::CmpLeS: case Op::CmpGtS: case Op::CmpGeS:
    case Op::CmpLtU: case Op::CmpLeU: case Op::CmpGtU: case Op::CmpGeU:
    case Op::FCmpEq: case Op::FCmpNe: case Op::FCmpLt:
    case Op::FCmpLe: case Op::FCmpGt: case Op::FCmpGe:
        return true;
    default:
        return false;
    }
}

bool Type::isSigned() const {
    
    return isInt();
}

std::string Type::str() const {
    std::ostringstream os;
    if (kind == TypeKind::Void) return "void";
    if (kind == TypeKind::Bool) os << "pred";
    else os << (kind == TypeKind::Int ? "i" : "f") << static_cast<int>(width);
    if (isVector()) os << "v" << static_cast<int>(components());
    return os.str();
}

const Function *Module::find(const std::string &n) const {
    for (const auto &f : functions)
        if (f.name == n) return &f;
    return nullptr;
}

Function *Module::find(const std::string &n) {
    for (auto &f : functions)
        if (f.name == n) return &f;
    return nullptr;
}

size_t Function::instrCount() const {
    size_t n = 0;
    for (const auto &b : blocks) n += b.instrs.size();
    return n;
}

namespace {

std::string operandStr(const Operand &o, ValueId self) {
    std::ostringstream os;
    switch (o.kind) {
    case Operand::Kind::Value:
        os << "%v" << o.value;
        break;
    case Operand::Kind::Immediate:
        os << o.imm;
        break;
    case Operand::Kind::GlobalSym:
        os << "@" << o.symbol;
        break;
    case Operand::Kind::ParamRef:
        os << o.symbol;
        break;
    case Operand::Kind::LabelRef:
        os << "bb" << o.block;
        break;
    }
    (void)self;
    return os.str();
}

std::string addrStr(const Address &a) {
    std::ostringstream os;
    os << "[";
    os << operandStr(a.base, kInvalidValue);
    if (a.hasIndex) os << " + " << operandStr(a.index, kInvalidValue) << "*" << a.scale;
    if (a.offset != 0) os << " + " << a.offset;
    os << "]";
    return os.str();
}

} 

std::string print(const Function &f) {
    std::ostringstream os;
    os << (f.isEntry ? ".entry " : ".func ") << f.name << "(";
    for (size_t i = 0; i < f.params.size(); ++i) {
        if (i) os << ", ";
        os << f.params[i].type.str() << " " << spaceName(f.params[i].space);
        if (!f.params[i].arrayDims.empty()) {
            os << "[";
            for (size_t d = 0; d < f.params[i].arrayDims.size(); ++d) {
                if (d) os << "][";
                os << f.params[i].arrayDims[d];
            }
            os << "]";
        }
    }
    os << ")";
    if (f.reqThreads) os << "  reqntid " << f.reqThreads;
    os << "\n{\n";

    for (size_t bi = 0; bi < f.blocks.size(); ++bi) {
        const auto &b = f.blocks[bi];
        
        
        os << "bb" << bi << ";";
        if (!b.label.empty()) os << "  // " << b.label;
        os << "\n";
        for (const auto &in : b.instrs) {
            if (in.hasResult()) os << "  %v" << in.result << " = ";
            os << opName(in.op);
            if (!in.type.isVoid()) os << " " << in.type.str();
            if (in.space != Space::Generic) os << " " << spaceName(in.space);
            for (const auto &m : in.mods) os << m;
            if (in.hasPred) os << " @%v" << in.pred;
            for (const auto &a : in.args) os << " " << operandStr(a, in.result);
            if (in.hasAddr) os << " " << addrStr(in.addr);
            os << "\n";
        }
    }
    os << "}\n";
    return os.str();
}

std::string print(const Module &m) {
    std::ostringstream os;
    os << "// module " << m.name << "\n";
    for (const auto &f : m.functions) os << print(f);
    for (const auto &n : m.notes) os << "// note: " << n << "\n";
    return os.str();
}

} 

//  MAIN: the lowering from PTX to the IR

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace ca::lower {
namespace {

using ir::Block;
using ir::Instr;
using ir::Operand;
using ir::Op;
using ir::Space;
using ir::Type;
using ir::TypeKind;
using ir::VecLen;

using ptx::Inst;
using ptx::OperandKind;

class LowerError : public std::runtime_error {
public:
    LowerError(const std::string &f, int line, const std::string &msg)
        : std::runtime_error(f + ":" + std::to_string(line) + ": " + msg) {}
};

Type mapType(const ptx::Type &t) {
    Type r;
    switch (t.kind) {
    case ptx::TypeKind::Void: r.kind = TypeKind::Void; return r;
    case ptx::TypeKind::Bool:
        r.kind = TypeKind::Bool;
        r.width = 1;
        return r;
    case ptx::TypeKind::Int:
        r.kind = TypeKind::Int;
        r.width = t.width;
        break;
    case ptx::TypeKind::Float:
        r.kind = TypeKind::Float;
        r.width = t.width;
        break;
    }
    if (t.vecLen == 2) r.vec = VecLen::V2;
    else if (t.vecLen == 4) r.vec = VecLen::V4;
    return r;
}

Space mapSpace(ptx::StateSpace s) {
    switch (s) {
    case ptx::StateSpace::Global: return Space::Global;
    case ptx::StateSpace::Shared: return Space::Shared;
    case ptx::StateSpace::Local: return Space::Local;
    case ptx::StateSpace::Constant: return Space::Constant;
    case ptx::StateSpace::Param: return Space::Param;
    case ptx::StateSpace::Generic: break;
    }
    return Space::Generic;
}

Space spaceFromMod(const Inst &in) {
    
    if (in.hasMod(".param")) return Space::Param;
    if (in.hasMod(".global")) return Space::Global;
    if (in.hasMod(".shared")) return Space::Shared;
    if (in.hasMod(".local")) return Space::Local;
    if (in.hasMod(".const")) return Space::Constant;
    return Space::Generic;
}

bool typeFromMods(const std::vector<std::string> &mods, Type &out) {
    static const char *const kTypes[] = {
        ".f16", ".f16x2", ".f32", ".f64", ".bf16", ".bf16x2",
        ".s8", ".s16", ".s32", ".s64",
        ".u8", ".u16", ".u32", ".u64",
        ".b8", ".b16", ".b32", ".b64", ".pred"};
    for (const char *m : kTypes) {
        const bool present = std::any_of(
            mods.begin(), mods.end(),
            [m](const std::string &s2) { return s2 == m; });
        if (!present) continue;
        
        auto widthOf = [](const char *p2) {
            unsigned w = 0;
            for (; *p2 && *p2 >= '0' && *p2 <= '9'; ++p2)
                w = w * 10 + static_cast<unsigned>(*p2 - '0');
            return w;
        };

        ptx::Type t;
        if (std::strcmp(m, ".pred") == 0) {
            t.kind = ptx::TypeKind::Bool;
            t.width = 1;
        } else if (m[1] == 'f') {
            t.kind = ptx::TypeKind::Float;
            t.width = widthOf(m + 2);
            if (std::strcmp(m, ".f16x2") == 0) t.vecLen = 2;
        } else if (m[1] == 'b' && m[2] == 'f') {
            t.kind = ptx::TypeKind::Float;
            t.width = widthOf(m + 3);
            if (std::strcmp(m, ".bf16x2") == 0) t.vecLen = 2;
        } else if (m[1] == 'b') {
            t.kind = ptx::TypeKind::Bool;
            t.width = widthOf(m + 2);
        } else {
            t.kind = ptx::TypeKind::Int;
            t.width = widthOf(m + 2);
        }
        out = mapType(t);
        return true;
    }
    return false;
}

Type resultType(const Inst &in, const ptx::Function &fn, const Type &fallback) {
    
    Type fromMod;
    if (typeFromMods(in.modifiers, fromMod)) return fromMod;

    
    if (!in.operands.empty() && in.operands[0].kind == OperandKind::Register) {
        if (const ptx::Type *rt = fn.registerType(in.operands[0].name))
            return mapType(*rt);
    }

    
    if (in.opcode.rfind("set", 0) == 0 || in.opcode.rfind("setp", 0) == 0) {
        return Type{TypeKind::Bool, 1, VecLen::Scalar};
    }

    return fallback;
}

struct Rule {
    const char *ptxOpcode; 
    Op op;
};

const Rule kIntRules[] = {
    {"add", Op::Add}, {"sub", Op::Sub}, {"mul", Op::Mul}, {"neg", Op::Neg},
    {"mov", Op::Mov}, {"cvt", Op::Cvt}, {"selp", Op::Selp},
    {"prefetch", Op::Unknown}, {"discard", Op::Unknown},
    {"rem", Op::RemS}, {"rem.u", Op::RemU}, {"shl", Op::Shl},
    {"shr.u", Op::ShrU}, {"and", Op::And}, {"or", Op::Or}, {"xor", Op::Xor},
    {"not", Op::Not},
    {"min", Op::MinS}, {"min.u", Op::MinU}, {"max", Op::MaxS}, {"max.u", Op::MaxU},
    {"mad", Op::FMulAdd}, {"mad24", Op::FMulAdd},
    {"mul.wide", Op::Mul}, {"mul.wide.u", Op::Mul},
    {"div", Op::DivS}, {"div.u", Op::DivU},
};

const Rule kFloatRules[] = {
    
    {"add", Op::FAdd}, {"sub", Op::FSub}, {"mul", Op::FMul}, {"div", Op::FDiv},
    {"neg.f32", Op::FNeg}, {"abs", Op::FAbs},
    {"abs.f32", Op::FAbs}, {"neg.f64", Op::FNeg}, {"abs.f64", Op::FAbs},
    {"min.f32", Op::MinF}, {"max.f32", Op::MaxF},
    {"fma.rn.f32", Op::FFma}, {"fma", Op::FFma},
    {"mad", Op::FMulAdd},
};

const Rule kMathRules[] = {
    {"ex2", Op::Exp2}, {"lg2", Op::Log2}, {"sqrt.rn.f32", Op::Sqrt},
    {"sqrt.rn.f64", Op::Sqrt}, {"rsqrt", Op::RSqrt}, {"sqrt.approx.f32", Op::Sqrt},
    {"sin", Op::Sin}, {"cos", Op::Cos}, {"tan", Op::Tan},
    {"pow", Op::Pow}, {"floor", Op::Floor}, {"ceil", Op::Ceil},
    {"rni", Op::Round}, {"cvt.rni.f32.f32", Op::Round},
    {"trunc", Op::Trunc}, {"cvt.rzi.f32.f32", Op::Trunc},
    {"rcp.rn.f32", Op::FDiv}, {"rcp.approx.f32", Op::FDiv},
    {"sin.approx.f32", Op::Sin}, {"cos.approx.f32", Op::Cos},
};

const Rule kSpecialRules[] = {
    {"%tid.x", Op::TidX}, {"%tid.y", Op::TidY}, {"%tid.z", Op::TidZ},
    {"%ctaid.x", Op::CtaidX}, {"%ctaid.y", Op::CtaidY}, {"%ctaid.z", Op::CtaidZ},
    {"%ntid.x", Op::NtidX}, {"%ntid.y", Op::NtidY}, {"%ntid.z", Op::NtidZ},
    {"%laneid", Op::LaneId}, {"%warpid", Op::WarpId}, {"%nwarpid", Op::WarpId},
    {"popc", Op::Popc}, {"vote.sync.all", Op::VoteAll},
    {"vote.sync.any", Op::VoteAny}, {"vote.sync.ballot", Op::VoteBallot},
    {"ballot.sync", Op::Ballot}, {"activemask", Op::ActiveMask},
    {"bar.sync", Op::BarrierThreads}, {"barrier.sync", Op::BarrierThreads},
    {"bar.warp.sync", Op::BarrierWarp}, {"barrier.warp.sync", Op::BarrierWarp},
    {"ret", Op::Ret}, {"ret.uni", Op::Ret},
    {"shfl.sync.idx", Op::ShflIdx}, {"shfl.sync.down", Op::ShflDown},
    {"shfl.sync.up", Op::ShflUp}, {"shfl.sync.bfly", Op::ShflXor},
    {"shfl.idx", Op::ShflIdx}, {"shfl.down", Op::ShflDown},
};

bool lookup(const Rule *table, size_t n, const std::string &op, Op &out) {
    for (size_t i = 0; i < n; ++i) {
        if (op == table[i].ptxOpcode) {
            out = table[i].op;
            return true;
        }
    }
    return false;
}

bool isFloatInst(const Inst &in) {
    static const char *const kFloats[] = {".f16", ".f32", ".f64", ".bf16",
                                          ".tf32", ".f16x2", ".f32x2"};
    for (const char *f : kFloats)
        if (in.hasMod(f)) return true;
    return false;
}

constexpr size_t kIntN = sizeof kIntRules / sizeof *kIntRules;
constexpr size_t kFloatN = sizeof kFloatRules / sizeof *kFloatRules;
constexpr size_t kMathN = sizeof kMathRules / sizeof *kMathRules;
constexpr size_t kSpecN = sizeof kSpecialRules / sizeof *kSpecialRules;

bool mapOpcode(const Inst &in, Op &out) {
    if (lookup(kSpecialRules, kSpecN, in.opcode, out)) return true;

    const bool flt = isFloatInst(in);
    if (flt) {
        if (lookup(kFloatRules, kFloatN, in.opcode, out)) return true;
        if (lookup(kMathRules, kMathN, in.opcode, out)) return true;
        if (lookup(kIntRules, kIntN, in.opcode, out)) return true;
    } else {
        if (lookup(kIntRules, kIntN, in.opcode, out)) return true;
        if (lookup(kMathRules, kMathN, in.opcode, out)) return true;
        if (lookup(kFloatRules, kFloatN, in.opcode, out)) return true;
    }

    const std::string head = in.opcode.substr(0, in.opcode.find('.'));
    if (head != in.opcode) {
        if (lookup(kSpecialRules, kSpecN, head, out)) return true;
        if (flt) {
            if (lookup(kFloatRules, kFloatN, head, out)) return true;
            if (lookup(kMathRules, kMathN, head, out)) return true;
        } else {
            if (lookup(kIntRules, kIntN, head, out)) return true;
            if (lookup(kMathRules, kMathN, head, out)) return true;
            if (lookup(kFloatRules, kFloatN, head, out)) return true;
        }
    }
    return false;
}

Op mapSetp(const Inst &in) {
    
    const std::string &o = in.opcode;
    auto has = [&](const char *m) { return in.hasMod(m); };
    const bool flt = in.hasMod(".f32") || in.hasMod(".f64") || in.hasMod(".f16");
    const bool uns = in.hasMod(".u32") || in.hasMod(".u64") || in.hasMod(".u16");

    if (o.rfind("setp", 0) == 0 || o.rfind("set", 0) == 0) {
        if (has(".eq")) return flt ? Op::FCmpEq : Op::CmpEq;
        if (has(".ne")) return flt ? Op::FCmpNe : Op::CmpNe;
        if (has(".lt")) return flt ? Op::FCmpLt : (uns ? Op::CmpLtU : Op::CmpLtS);
        if (has(".le")) return flt ? Op::FCmpLe : (uns ? Op::CmpLeU : Op::CmpLeS);
        if (has(".gt")) return flt ? Op::FCmpGt : (uns ? Op::CmpGtU : Op::CmpGtS);
        if (has(".ge")) return flt ? Op::FCmpGe : (uns ? Op::CmpGeU : Op::CmpGeS);
    }
    return Op::Unknown;
}

class FuncLowerer {
public:
    FuncLowerer(const ptx::Function &fn, const Options &opt, Result &res)
        : fn_(fn), opt_(opt), res_(res) {}

    ir::Function run();

private:
    const ptx::Function &fn_;
    const Options &opt_;
    Result &res_;

    ir::Function out_;
    ir::ValueId nextValue_ = 0;

    
    std::unordered_map<std::string, ir::ValueId> regs_;
    std::unordered_map<std::string, ir::ValueId> undeclared_;
    
    std::unordered_map<std::string, ir::BlockId> labels_;

    ir::Block *cur_ = nullptr;

    ir::ValueId alloc() { return nextValue_++; }

    
    void emitUnknown(const ptx::Inst &in);

    Operand convertOperand(const ptx::Operand &o, const Type &hint);

    
    Operand materialize(const ptx::Operand &o, const Type &hint);

    
    ir::ValueId freshUndeclared(const std::string &regName);
    ir::Address convertAddress(const ptx::Operand &o);

    void note(const std::string &msg) {
        res_.module.notes.push_back(fn_.name + ": " + msg);
    }

    void countIn(const Instr &in) {
        if (in.op == Op::Unknown) {
            ++res_.unknown;
        } else {
            ++res_.lowered;
            if (ir::isWarpOp(in.op)) ++res_.warpOps;
        }
    }
};

ir::Address FuncLowerer::convertAddress(const ptx::Operand &o) {
    ir::Address a;
    if (o.kind != OperandKind::Addr) return a;

    if (o.addr.isParamRef) {
        a.base = Operand::param(o.addr.baseReg);
    } else {
        auto it = regs_.find(o.addr.baseReg);
        if (it != regs_.end()) a.base = Operand::val(it->second);
        else a.base = Operand::sym(o.addr.baseReg);
    }
    a.offset = o.addr.offset;
    if (o.addr.hasIndex) {
        auto it = regs_.find(o.addr.indexReg);
        a.hasIndex = true;
        a.index = (it != regs_.end()) ? Operand::val(it->second)
                                      : Operand::sym(o.addr.indexReg);
        
        a.scale = o.addr.scale ? o.addr.scale : 1;
    }
    return a;
}

ir::ValueId FuncLowerer::freshUndeclared(const std::string &regName) {
    
    
    
    auto it = undeclared_.find(regName);
    if (it != undeclared_.end()) return it->second;
    const ir::ValueId v = alloc();
    undeclared_[regName] = v;
    regs_[regName] = v;
    out_.declared.push_back(v);
    return v;
}

Operand FuncLowerer::materialize(const ptx::Operand &o, const Type &hint) {
    if (o.kind == OperandKind::Register) {
        struct Spec {
            const char *name;
            Op op;
        };
        static const Spec kSpec[] = {
            {"tid.x", Op::TidX},   {"tid.y", Op::TidY},   {"tid.z", Op::TidZ},
            {"ctaid.x", Op::CtaidX}, {"ctaid.y", Op::CtaidY}, {"ctaid.z", Op::CtaidZ},
            {"ntid.x", Op::NtidX}, {"ntid.y", Op::NtidY}, {"ntid.z", Op::NtidZ},
            {"laneid", Op::LaneId}, {"warpid", Op::WarpId},
        };
        for (const auto &s : kSpec) {
            if (o.name != s.name) continue;
            Instr si;
            si.op = s.op;
            si.type = Type{TypeKind::Int, 32, VecLen::Scalar};
            si.result = alloc();
            si.line = 0;
            const ir::ValueId v = si.result;
            cur_->instrs.push_back(std::move(si));
            ++res_.lowered;
            return Operand::val(v);
        }
    }
    return convertOperand(o, hint);
}

Operand FuncLowerer::convertOperand(const ptx::Operand &o, const Type &hint) {
    switch (o.kind) {
    case OperandKind::Immediate: {
        Operand r = Operand::lit(o.imm);
        r.immBits = static_cast<uint64_t>(o.imm);
        return r;
    }
    case OperandKind::Register: {
        auto it = regs_.find(o.name);
        if (it != regs_.end()) return Operand::val(it->second);
        return Operand::val(freshUndeclared(o.name));
    }
    case OperandKind::Param:
        return Operand::param(o.name);
    case OperandKind::Label:
        return Operand::sym(o.name);
    case OperandKind::Addr:
        return Operand::sym("[addr]");
    case OperandKind::Vector: {
        
        if (o.elements.size() == 1) return convertOperand(o.elements[0], hint);
        Operand r = convertOperand(o.elements[0], hint);
        return r;
    }
    case OperandKind::Symbol:
    case OperandKind::Global:
    case OperandKind::Constant:
    case OperandKind::Local:
    case OperandKind::FuncRef:
    case OperandKind::TypeRef:
        return Operand::sym(o.name);
    case OperandKind::Special:
        return Operand::sym(o.name);
    }
    return Operand::sym("");
}

void FuncLowerer::emitUnknown(const ptx::Inst &in) {
    Instr ir_in;
    ir_in.op = Op::Unknown;
    ir_in.type = Type{};
    ir_in.line = in.line;

    
    std::string txt = in.opcode;
    for (const auto &m : in.modifiers) txt += m;
    for (const auto &o : in.operands) txt += " " + o.str();
    ir_in.origText = txt;

    if (opt_.strict)
        throw LowerError(fn_.name, in.line, "unsupported instruction: " + txt);

    
    if (!in.operands.empty() && in.operands[0].kind == OperandKind::Register) {
        ir_in.result = regs_[in.operands[0].name];
        ir_in.type = resultType(in, fn_, Type{TypeKind::Int, 32, VecLen::Scalar});
        
        for (size_t k = 1; k < in.operands.size(); ++k)
            ir_in.args.push_back(convertOperand(in.operands[k], ir_in.type));
    } else {
        ir_in.result = ir::kInvalidValue;
    }

    
    countIn(ir_in);
    cur_->instrs.push_back(std::move(ir_in));
}

ir::Function FuncLowerer::run() {
    out_.name = fn_.name;
    out_.isEntry = fn_.visible;
    out_.reqThreads = fn_.maxThreads;

    for (const auto &p : fn_.params) {
        ir::Param ip;
        ip.name = p.name;
        ip.type = mapType(p.type);
        ip.space = mapSpace(p.type.space);
        ip.isPointer = p.isPointer;
        ip.arrayDims = p.arrayDims;
        out_.params.push_back(std::move(ip));
    }

    
    for (const auto &rd : fn_.regs) {
        for (unsigned i = 0; i < rd.slots(); ++i) {
            const ir::ValueId v = alloc();
            regs_[rd.name + std::to_string(i)] = v;
            
            out_.declared.push_back(v);
        }
    }

    
    Block first;
    first.label = "entry";
    out_.blocks.push_back(std::move(first));
    cur_ = &out_.blocks[0];

    

    auto isBranch = [](const ptx::Inst &i) {
        return i.opcode == "bra.uni" || i.opcode == "bra" || i.opcode == "br";
    };

    std::unordered_map<std::string, ir::BlockId> labelToBlock;
    std::unordered_map<size_t, ir::BlockId> blockAfterBranch;
    ir::BlockId nextId = 1;
    ir::BlockId curId = 0;
    bool empty = true;
    for (size_t i = 0; i < fn_.body.size(); ++i) {
        const auto &pin = fn_.body[i];
        if (pin.isLabel) {
            if (!empty) {
                curId = nextId++;
                empty = true;
            }
            if (!labelToBlock.count(pin.label)) labelToBlock[pin.label] = curId;
            continue;
        }
        empty = false;
        if (isBranch(pin)) {
            curId = nextId++;
            empty = true;
            blockAfterBranch[i] = curId;
        }
    }
    const ir::BlockId blockCount = nextId;

    out_.blocks.resize(blockCount);
    out_.blocks[0].label = "entry";
    for (const auto &kv : labelToBlock)
        if (kv.second < blockCount && out_.blocks[kv.second].label.empty())
            out_.blocks[kv.second].label = kv.first;

    cur_ = &out_.blocks[0];

    for (const auto &pin : fn_.body) {
        const ptx::Inst &pi = pin;

        if (pi.isLabel) {
            
            auto it = labelToBlock.find(pi.label);
            if (it != labelToBlock.end() && it->second < out_.blocks.size())
                cur_ = &out_.blocks[it->second];
            continue;
        }

        Op op = Op::Unknown;
        if (!mapOpcode(pi, op)) {
            const Op s = mapSetp(pi);
            if (s != Op::Unknown) op = s;
        }
        // The atomics carry their operation in the modifiers: atom.add, atom.cas, red.max.
        if (op == Op::Unknown && (pi.opcode == "atom" || pi.opcode == "red")) {
            const bool compareAndSwap =
                std::any_of(pi.modifiers.begin(), pi.modifiers.end(),
                            [](const std::string &m) { return m == ".cas"; });
            op = compareAndSwap ? Op::AtomicCAS : Op::AtomicRMW;
        }

        
        if (op == Op::Ret) {
            Instr t;
            t.op = op;
            t.type = Type{};
            t.result = ir::kInvalidValue;
            t.line = pi.line;
            countIn(t);
            cur_->instrs.push_back(std::move(t));
            continue;
        }

        if (pi.opcode == "bra.uni" || pi.opcode == "bra" || pi.opcode == "br") {
            Instr br;
            br.op = Op::CondBr;
            br.type = Type{};
            br.result = ir::kInvalidValue;
            br.line = pi.line;
            if (pi.hasPred) {
                auto it = regs_.find(pi.pred);
                br.hasPred = true;
                br.pred = (it != regs_.end()) ? it->second : freshUndeclared(pi.pred);
            }
            if (!pi.operands.empty() && pi.operands[0].kind == OperandKind::Label) {
                auto it = labelToBlock.find(pi.operands[0].name);
                if (it != labelToBlock.end()) {
                    br.args.push_back(Operand::branch(it->second));
                } else {
                    
                    br.args.push_back(Operand::sym(pi.operands[0].name));
                }
            }
            countIn(br);
            cur_->instrs.push_back(std::move(br));

            
            auto nb = blockAfterBranch.find(
                static_cast<size_t>(&pi - fn_.body.data()));
            if (nb != blockAfterBranch.end() && nb->second < out_.blocks.size())
                cur_ = &out_.blocks[nb->second];
            continue;
        }

        Instr in;
        in.op = op;
        in.line = pi.line;
        in.mods = pi.modifiers;
        in.space = spaceFromMod(pi);

        if (pi.hasPred) {
            auto it = regs_.find(pi.pred);
            
            in.hasPred = true;
            in.pred = (it != regs_.end()) ? it->second : freshUndeclared(pi.pred);
            if (!pi.pred.empty() && pi.pred[0] == '!') {
                Instr neg;
                neg.op = Op::Not;
                neg.type = Type{TypeKind::Bool, 1, VecLen::Scalar};
                neg.result = alloc();
                neg.args.push_back(Operand::val(in.pred));
                cur_->instrs.push_back(std::move(neg));
                in.pred = neg.result;
            }
        }

        
        const bool isAtomic = op == Op::AtomicRMW || op == Op::AtomicCAS;
        const bool isMemOp = pi.opcode == "ld" || pi.opcode == "st" || isAtomic;

        if (isMemOp) {
            const bool isLoad = (pi.opcode == "ld");
            if (!isAtomic) in.op = isLoad ? Op::Load : Op::Store;
            in.hasAddr = true;
            in.space = spaceFromMod(pi);

            
            size_t addrIdx = SIZE_MAX;
            for (size_t k = 0; k < pi.operands.size(); ++k)
                if (pi.operands[k].kind == OperandKind::Addr) {
                    addrIdx = k;
                    break;
                }
            if (addrIdx != SIZE_MAX) {
                in.addr = convertAddress(pi.operands[addrIdx]);
            }

            const bool hasDestination = isLoad || isAtomic;
            for (size_t k = 0; k < pi.operands.size(); ++k) {
                if (k == addrIdx) continue;
                if (hasDestination && k == 0) continue;
                if (!isLoad && !isAtomic && k == 0 && addrIdx != SIZE_MAX) continue;
                in.args.push_back(convertOperand(pi.operands[k], Type{}));
            }

            
            Type elem;
            if (!typeFromMods(pi.modifiers, elem))
                elem = Type{TypeKind::Int, 32, VecLen::Scalar};
            auto hasMod = [&](const char *m) {
                return std::any_of(pi.modifiers.begin(), pi.modifiers.end(),
                                   [m](const std::string &x) { return x == m; });
            };
            if (hasMod(".v4")) elem.vec = VecLen::V4;
            else if (hasMod(".v2")) elem.vec = VecLen::V2;

            if (in.op == Op::Load || isAtomic) {
                if (!pi.operands.empty() &&
                    pi.operands[0].kind == OperandKind::Register)
                    in.result = regs_[pi.operands[0].name];
                else if (isAtomic)
                    in.result = ir::kInvalidValue; // red: no result
                else
                    in.result = alloc();
                in.type = elem;
            } else {
                in.type = elem;
                in.result = ir::kInvalidValue;
            }
            countIn(in);
            cur_->instrs.push_back(std::move(in));
            continue;
        }

        
        if (in.op == Op::Unknown) {
            in.pred = in.hasPred ? in.pred : ir::kInvalidValue;
            emitUnknown(pi);
            continue;
        }

        
        {
            if (isComparison(in.op)) {
                in.type = Type{TypeKind::Bool, 1, VecLen::Scalar};
            } else if (ir::isFloatOp(in.op)) {
                in.type = resultType(pi, fn_, Type{TypeKind::Float, 32, VecLen::Scalar});
            } else {
                in.type = resultType(pi, fn_, Type{TypeKind::Int, 32, VecLen::Scalar});
            }

            const bool noResult = in.op == Op::Unknown &&
                                  (pi.opcode == "prefetch" ||
                                   pi.opcode == "discard");
            if (!noResult) {
                if (!pi.operands.empty() &&
                    pi.operands[0].kind == OperandKind::Register &&
                    regs_.count(pi.operands[0].name))
                    in.result = regs_[pi.operands[0].name];
                else
                    in.result = alloc();
            } else {
                in.result = ir::kInvalidValue;
                in.type = Type{};
            }

            
            size_t firstArg = 0;
            if (!pi.operands.empty() &&
                pi.operands[0].kind == OperandKind::Register)
                firstArg = 1;
            for (size_t k = firstArg; k < pi.operands.size(); ++k)
                in.args.push_back(materialize(pi.operands[k], in.type));

            
            if (in.op == Op::Cvt) {
                for (const auto &m : pi.modifiers) {
                    Type t;
                    if (typeFromMods({m}, t) && !t.isVoid()) {
                        in.type = t;
                        break;
                    }
                }
            }
        }
        countIn(in);
        cur_->instrs.push_back(std::move(in));
    }

    
    
    
    
    
    for (size_t i = 0; i < out_.blocks.size(); ++i) {
        Block &b = out_.blocks[i];
        if (b.instrs.empty()) continue;
        if (ir::isTerminator(b.instrs.back().op)) continue;
        if (i + 1 < out_.blocks.size()) continue;
        Instr r;
        r.op = Op::Ret;
        b.instrs.push_back(std::move(r));
    }
    if (out_.blocks.empty() || out_.blocks[0].instrs.empty()) {
        Block b;
        b.label = "entry";
        Instr r;
        r.op = Op::Ret;
        b.instrs.push_back(std::move(r));
        out_.blocks.insert(out_.blocks.begin(), std::move(b));
    }

    return std::move(out_);
}

} 

Result run(const ptx::Module &in, const Options &opt) {
    Result res;
    res.module.name = "cuascend";
    for (const auto &f : in.functions) {
        if (f.external) continue; 
        FuncLowerer fl(f, opt, res);
        res.module.functions.push_back(fl.run());
    }
    return res;
}

Result fromText(const std::string &ptxText, const Options &opt) {
    Result r;
    r.bytes = ptxText.size();
    ptx::Module pm;
    try {
        pm = ptx::parse(ptxText);
    } catch (const ptx::Error &e) {
        // A malformed image is data, not a reason to take the process down: the caller gets an empty module and a note.
        r.module.notes.push_back(std::string("PTX parse error: ") + e.what());
        return r;
    }
    r = run(pm, opt);
    r.bytes = ptxText.size();
    return r;
}

} 

// MAIN: the IR verifier

#include <sstream>
#include <unordered_set>

namespace ca::ir {
namespace {

std::string at(const Function &f) {
    return " in function '" + f.name + "'";
}

std::string where(const Function &f, const Block &b, size_t idx) {
    std::ostringstream os;
    os << "block " << (&b - &f.blocks[0]) << " (" << b.label << "), instr #"
       << idx << at(f);
    return os.str();
}

} 

std::vector<std::string> verify(const Module &m) {
    std::vector<std::string> problems;

    for (const auto &f : m.functions) {
        if (f.blocks.empty()) {
            problems.push_back("function '" + f.name + "' has no blocks");
            continue;
        }

        
        std::unordered_set<ValueId> defined;
        for (ValueId v : f.declared) defined.insert(v);
        for (const auto &b : f.blocks)
            for (const auto &in : b.instrs)
                if (in.hasResult()) defined.insert(in.result);

        for (size_t bi = 0; bi < f.blocks.size(); ++bi) {
            const auto &b = f.blocks[bi];
            if (b.instrs.empty()) {
                problems.push_back("empty block " + b.label + at(f));
                continue;
            }

            for (size_t i = 0; i < b.instrs.size(); ++i) {
                const auto &in = b.instrs[i];

                
                if (isTerminator(in.op) && i + 1 != b.instrs.size()) {
                    problems.push_back(std::string("terminator '") + opName(in.op) +
                                       "' is not last in " + where(f, b, i));
                }

                
                if (i > 0 && isTerminator(b.instrs[i - 1].op)) {
                    problems.push_back(std::string("instruction '") + opName(in.op) +
                                       "' follows a terminator in " +
                                       where(f, b, i));
                }

                
                auto checkRef = [&](const Operand &o, const char *what) {
                    if (o.isValue() && !defined.count(o.value)) {
                        problems.push_back(std::string(what) + " references undefined "
                                           "value %v" +
                                           std::to_string(o.value) + " in " +
                                           where(f, b, i));
                    }
                    if (o.isBranch() && o.block >= f.blocks.size()) {
                        problems.push_back(std::string(what) + " branches to bad block "
                                           "bb" + std::to_string(o.block) + " in " +
                                           where(f, b, i));
                    }
                };

                for (size_t k = 0; k < in.args.size(); ++k)
                    checkRef(in.args[k], "argument");

                if (in.hasAddr) {
                    checkRef(in.addr.base, "address base");
                    if (in.addr.hasIndex) checkRef(in.addr.index, "address index");
                }

                
                if (in.hasPred && !defined.count(in.pred)) {
                    problems.push_back("undefined predicate %v" +
                                       std::to_string(in.pred) + " in " +
                                       where(f, b, i));
                }

                
                if ((in.op == Op::Load || in.op == Op::Store) && !in.hasAddr) {
                    problems.push_back(std::string(opName(in.op)) +
                                       " without address in " + where(f, b, i));
                }

                
                if (in.hasResult() && in.type.isVoid()) {
                    problems.push_back(std::string(opName(in.op)) +
                                       " defines a value but has void type in " +
                                       where(f, b, i));
                }

                
                if (isComparison(in.op) && in.hasResult() && !in.type.isBool()) {
                    problems.push_back(std::string(opName(in.op)) +
                                       " must produce pred, got " + in.type.str() +
                                       " in " + where(f, b, i));
                }

                
                if (in.op == Op::Store && in.hasResult()) {
                    problems.push_back("store defines a value in " + where(f, b, i));
                }
            }

            
            
            if (!b.terminated() && bi + 1 == f.blocks.size()) {
                problems.push_back("block '" + b.label + "' has no terminator" +
                                   at(f));
            }
        }
    }

    return problems;
}

} 

//  MAIN: the AscendC generator

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>

namespace ca::codegen {
namespace {

using ir::Instr;
using ir::Op;
using ir::Operand;
using ir::Type;
using ir::TypeKind;
using ir::ValueId;
using ir::VecLen;

std::string valName(ValueId v) { return "v" + std::to_string(v); }

std::string paramName(size_t i) {
    return "tiling_data.arg" + std::to_string(i);
}

const char *cType(const Type &t) {
    if (t.isBool()) return "bool";
    switch (t.width) {
    case 8: return t.isFloat() ? "half" : (t.isInt() ? "int8_t" : "uint8_t");
    case 16: return t.isFloat() ? "half" : (t.isInt() ? "int16_t" : "uint16_t");
    case 32: return t.isFloat() ? "float" : (t.isInt() ? "int32_t" : "uint32_t");
    case 64: return t.isFloat() ? "double" : (t.isInt() ? "int64_t" : "uint64_t");
    default: break;
    }
    return "float";
}

bool canLower(Op op) {
    switch (op) {
    case Op::Load: case Op::Store:
    case Op::Add: case Op::Sub: case Op::Mul:
    case Op::DivS: case Op::DivU: case Op::RemS: case Op::RemU:
    case Op::Neg: case Op::Not:
    case Op::And: case Op::Or: case Op::Xor:
    case Op::Shl: case Op::ShrU:
    case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv:
    case Op::FNeg: case Op::FAbs: case Op::FFma:
    case Op::CmpEq: case Op::CmpNe:
    case Op::CmpLtS: case Op::CmpLtU:
    case Op::CmpLeS: case Op::CmpLeU:
    case Op::CmpGtS: case Op::CmpGtU:
    case Op::CmpGeS: case Op::CmpGeU:
    case Op::FCmpEq: case Op::FCmpNe: case Op::FCmpLt:
    case Op::FCmpLe: case Op::FCmpGt: case Op::FCmpGe:
    case Op::Sqrt: case Op::RSqrt:
    case Op::Exp: case Op::Exp2: case Op::Log: case Op::Log2:
    case Op::Floor: case Op::Ceil: case Op::Round: case Op::Trunc:
    case Op::Sin: case Op::Cos: case Op::Pow:
    case Op::Cvt: case Op::Selp: case Op::Mov:
    case Op::BarrierThreads:
    case Op::TidX: case Op::CtaidX: case Op::LaneId: case Op::WarpId:
    case Op::NtidX: case Op::NtidY: case Op::NtidZ:
    case Op::Popc: case Op::NumWarps:
    
    case Op::Ret:
    case Op::CondBr:
        return true;
    default:
        return false;
    }
}

const char *whyNot(Op op) {
    switch (op) {
    case Op::ShrS:
        return "arithmetic shift needs the operand's signedness, which IR "
               "does not carry; a sign flag is required";
    case Op::MinS: case Op::MinU: case Op::MaxS: case Op::MaxU:
    case Op::MinF: case Op::MaxF:
        return "min/max are ambiguous for IEEE-754 on NaN and on 0.0/-0.0; "
               "the semantics need an explicit decision, not a default";
    case Op::FMulAdd:
        return "fmuladd has two rounding modes depending on contraction; "
               "that cannot be decided by default";
    case Op::ShflIdx: case Op::ShflUp: case Op::ShflDown: case Op::ShflXor:
    case Op::ShflPred:
        return "lane exchange has no counterpart once SIMT is unrolled: the "
               "warp no longer exists. A pass that keeps warps together is "
               "required, not a per-element substitute";
    case Op::VoteAll: case Op::VoteAny: case Op::VoteBallot:
    case Op::Ballot: case Op::ActiveMask:
        return "warp voting has no counterpart once SIMT is unrolled — same "
               "reason as exchange";
    case Op::BarrierWarp:
        return "a warp barrier has no counterpart: after unrolling there is "
               "no warp";
    case Op::AtomicRMW: case Op::AtomicCAS:
        return "atomics need a memory-model decision that IR does not carry "
               "yet";
    case Op::Br:
        return "an unconditional branch is not represented in the unrolled "
               "model: a loop needs a structural transformation";
    case Op::CondBr:
        return "conditional branches are supported only as a guard exit "
               "(@p bra SKIP); anything else needs a structural "
               "transformation";
    case Op::Call:
        return "function calls are not lowered";
    case Op::Unknown:
        return "the operation was not recognised while parsing PTX; its "
               "source text is kept in origText";
    case Op::TidY: case Op::TidZ: case Op::CtaidY: case Op::CtaidZ:
        return "y/z coordinates are not represented by the linear "
               "unrolling";
    default:
        break;
    }
    return "no AscendC mapping is defined";
}

class Gen {
public:
    Gen(const ir::Function &fn, const TargetInfo &target)
        : fn_(fn), target_(target) {}

    Output run();

private:
    const ir::Function &fn_;
    const TargetInfo target_;

    
    std::ostringstream hoist_;
    std::ostringstream decls_;
    std::ostringstream body_;
    std::map<ValueId, Type> types_;

    
    int indent_ = 2;

    void line(const std::string &s) {
        body_ << std::string(indent_ * 4, ' ') << s << "\n";
    }

    [[noreturn]] void reject(const Instr &in, const char *detail = nullptr) {
        std::string why = whyNot(in.op);
        if (detail) {
            std::string d = detail;
            
            why += " (";
            if (in.line) why += "PTX line " + std::to_string(in.line) + ": ";
            why += d;
            why += ")";
        }
        throw CodegenError(fn_.name, ir::opName(in.op), why);
    }

    
    void openIf(const Instr &in) {
        if (in.hasPred) {
            line("if (" + valName(in.pred) + ") {");
            ++indent_;
        }
    }
    void closeIf(const Instr &in) {
        if (in.hasPred) {
            --indent_;
            line("}");
        }
    }

    void assign(const Instr &in, const std::string &expr);
    
    std::string paramSymbol(const std::string &s);

    
    std::map<std::string, std::string> paramMap_;
    std::string operand(const Operand &o, const Type &hint);
    Type typeOf(ValueId v) const;

    void emitLoad(const Instr &in);
    void emitStore(const Instr &in);
    void emitArith(const Instr &in);
    void emitFloat(const Instr &in);
    void emitIntCmp(const Instr &in);
    void emitFloatCmp(const Instr &in);
    void emitUnary(const Instr &in);
    void emitMath(const Instr &in);
    void emitSpecial(const Instr &in);
    void emitCvt(const Instr &in);
    void emitSelp(const Instr &in);
    void emitMove(const Instr &in);

    
    Type memType(const Instr &in) const;
};

std::string Gen::paramSymbol(const std::string &s) {
    if (!s.empty() && s[0] == '$')
        return paramName(
            static_cast<size_t>(std::strtoul(s.c_str() + 1, nullptr, 10)));
    auto it = paramMap_.find(s);
    return it == paramMap_.end() ? s : it->second;
}

Type Gen::typeOf(ValueId v) const {
    auto it = types_.find(v);
    return it == types_.end() ? Type{TypeKind::Float, 32, VecLen::Scalar}
                              : it->second;
}

void Gen::assign(const Instr &in, const std::string &expr) {
    if (!in.hasResult()) {
        line(expr + ";");
        return;
    }
    openIf(in);
    line(valName(in.result) + " = " + expr + ";");
    closeIf(in);
}

std::string Gen::operand(const Operand &o, const Type &hint) {
    switch (o.kind) {
    case Operand::Kind::Value:
        return valName(o.value);
    case Operand::Kind::ParamRef:
        return paramSymbol(o.symbol);
    case Operand::Kind::GlobalSym:
        throw CodegenError(fn_.name, "symbol '" + o.symbol + "'",
                           "external symbol tables are not resolved against "
                           "AscendC buffers");
    case Operand::Kind::LabelRef:
        throw CodegenError(fn_.name, "label reference",
                           "control flow is not lowered");
    case Operand::Kind::Immediate: {
        if (hint.isFloat()) {
            char buf[64];
            if (hint.width == 32) {
                float f;
                std::memcpy(&f, &o.immBits, sizeof f);
                if (f == static_cast<float>(static_cast<long long>(f)))
                    std::snprintf(buf, sizeof buf, "%lldLL",
                                  static_cast<long long>(f));
                else
                    std::snprintf(buf, sizeof buf, "%.9gf",
                                  static_cast<double>(f));
                return buf;
            }
            if (hint.width == 64) {
                double d;
                std::memcpy(&d, &o.immBits, sizeof d);
                std::snprintf(buf, sizeof buf, "%.17g", d);
                return buf;
            }
            std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(o.immBits));
            return buf;
        }
        if (hint.isBool()) return o.imm ? "true" : "false";
        
        if (hint.width == 64) return std::to_string(o.imm) + "LL";
        return std::to_string(o.imm);
    }
    }
    return "0";
}

Type Gen::memType(const Instr &in) const {
    
    if (!in.type.isVoid()) return in.type;
    if (!in.args.empty() && in.args[0].isValue()) return typeOf(in.args[0].value);
    return Type{TypeKind::Float, 32, VecLen::Scalar};
}

void Gen::emitLoad(const Instr &in) {
    if (!in.hasAddr) reject(in);

    
    if (in.space == ir::Space::Param) {
        const Type t = memType(in);
        const std::string tn = cType(t);
        const Operand &base = in.addr.base;
        if (base.kind != Operand::Kind::ParamRef)
            reject(in, "ld.param without a parameter as its address base");
        const std::string baseExpr = paramSymbol(base.symbol);
        openIf(in);
        line(valName(in.result) + " = static_cast<" + tn + ">(" + baseExpr + ");");
        closeIf(in);
        return;
    }

    if (in.space == ir::Space::Shared)
        throw CodegenError(fn_.name, "load from shared",
                           "CUDA shared memory is not shared between Ascend cores; "
                           "mapping it to a LocalBuffer needs a placement model "
                           "that IR does not have yet");
    const Type t = memType(in);
    const std::string tn = cType(t);
    
    const Operand &base = in.addr.base;
    std::string baseExpr;
    if (base.isValue()) baseExpr = valName(base.value);
    else if (base.kind == Operand::Kind::ParamRef) baseExpr = paramSymbol(base.symbol);
    else reject(in, "address base is neither a value nor a parameter");

    
    const std::string tensor = "gt_" + tn + "_" + std::to_string(in.result);
    const std::string local = "read_" + tn + "_" + std::to_string(in.result);
    openIf(in);
    line("{");
    line("    GlobalTensor<" + tn + "> " + tensor + ";");
    line("    " + tensor + ".SetGlobalBuffer(reinterpret_cast<__gm__ " + tn +
         "*>(" + baseExpr + "), totalThreads);");
    if (in.hasResult())
        line("    " + valName(in.result) + " = " + tensor + ".GetValue(elemIndex);");
    line("}");
    closeIf(in);
}

void Gen::emitStore(const Instr &in) {
    if (!in.hasAddr) reject(in);
    if (in.space == ir::Space::Param) {
        
        reject(in);
    }
    if (in.space == ir::Space::Shared)
        throw CodegenError(fn_.name, "store to shared",
                           "CUDA shared memory is not shared between Ascend cores; "
                           "mapping it to a LocalBuffer needs a placement model "
                           "that IR does not have yet");
    if (in.args.size() != 1) reject(in);
    const Type t = in.args[0].isValue() ? typeOf(in.args[0].value) : memType(in);
    const std::string tn = cType(t);

    const Operand &base = in.addr.base;
    std::string baseExpr;
    if (base.isValue()) baseExpr = valName(base.value);
    else if (base.kind == Operand::Kind::ParamRef) baseExpr = paramSymbol(base.symbol);
    else reject(in);

    const std::string tensor = "st_" + tn + "_" + std::to_string(in.line);
    openIf(in);
    line("{");
    ++indent_;
    line("GlobalTensor<" + tn + "> " + tensor + ";");
    line(tensor + ".SetGlobalBuffer(reinterpret_cast<__gm__ " + tn + "*>(" +
         baseExpr + "), totalThreads);");
    line(tensor + ".SetValue(elemIndex, " + operand(in.args[0], t) + ");");
    --indent_;
    line("}");
    closeIf(in);
}

void Gen::emitArith(const Instr &in) {
    if (in.args.size() != 2) reject(in);
    const Type &t = in.type;
    static const std::map<Op, const char *> kOps = {
        {Op::Add, "+"},   {Op::Sub, "-"},   {Op::Mul, "*"},
        {Op::DivS, "/"},  {Op::DivU, "/"},  {Op::RemS, "%"},
        {Op::RemU, "%"},  {Op::And, "&"},   {Op::Or, "|"},
        {Op::Xor, "^"},   {Op::Shl, "<<"},  {Op::ShrU, ">>"},
    };
    auto it = kOps.find(in.op);
    if (it == kOps.end()) reject(in);
    const std::string a = operand(in.args[0], t);
    const std::string b = operand(in.args[1], t);
    assign(in, "((" + a + ") " + it->second + " (" + b + "))");
}

void Gen::emitFloat(const Instr &in) {
    if (in.args.size() != 2) reject(in);
    const Type &t = in.type;
    static const std::map<Op, const char *> kOps = {
        {Op::FAdd, "+"}, {Op::FSub, "-"}, {Op::FMul, "*"}, {Op::FDiv, "/"}};
    auto it = kOps.find(in.op);
    if (it == kOps.end()) reject(in);
    assign(in, "((" + operand(in.args[0], t) + ") " + it->second + " (" +
                   operand(in.args[1], t) + "))");
}

void Gen::emitUnary(const Instr &in) {
    if (in.args.size() != 1) reject(in);
    const Type &t = in.type;
    const std::string a = operand(in.args[0], t);
    switch (in.op) {
    case Op::Neg: case Op::FNeg: assign(in, "(-(" + a + "))"); return;
    case Op::Not: assign(in, "(~(" + a + "))"); return;
    case Op::FAbs: assign(in, "(fabs(" + a + "))"); return;
    default: reject(in);
    }
}

void Gen::emitIntCmp(const Instr &in) {
    if (in.args.size() != 2) reject(in);
    
    Type t = in.args[0].isValue() ? typeOf(in.args[0].value)
                                  : Type{TypeKind::Int, 32, VecLen::Scalar};
    if (t.isVoid()) t = Type{TypeKind::Int, 32, VecLen::Scalar};
    const std::string a = operand(in.args[0], t);
    const std::string b = operand(in.args[1], t);
    const char *o = "==";
    switch (in.op) {
    case Op::CmpEq: o = "=="; break;
    case Op::CmpNe: o = "!="; break;
    case Op::CmpLtS: case Op::CmpLtU: o = "<"; break;
    case Op::CmpLeS: case Op::CmpLeU: o = "<="; break;
    case Op::CmpGtS: case Op::CmpGtU: o = ">"; break;
    case Op::CmpGeS: case Op::CmpGeU: o = ">="; break;
    default: reject(in);
    }
    assign(in, "((" + a + ") " + o + " (" + b + "))");
}

void Gen::emitFloatCmp(const Instr &in) {
    if (in.args.size() != 2) reject(in);
    Type t = in.args[0].isValue() ? typeOf(in.args[0].value)
                                  : Type{TypeKind::Float, 32, VecLen::Scalar};
    if (!t.isFloat()) t = Type{TypeKind::Float, 32, VecLen::Scalar};
    const std::string a = operand(in.args[0], t);
    const std::string b = operand(in.args[1], t);
    const char *o = "==";
    switch (in.op) {
    case Op::FCmpEq: o = "=="; break;
    case Op::FCmpNe: o = "!="; break;
    case Op::FCmpLt: o = "<"; break;
    case Op::FCmpLe: o = "<="; break;
    case Op::FCmpGt: o = ">"; break;
    case Op::FCmpGe: o = ">="; break;
    default: reject(in);
    }
    
    assign(in, "(fabs(" + a + ") " + o + " fabs(" + b + "))");
}

void Gen::emitMath(const Instr &in) {
    if (in.args.size() != 1) reject(in);
    const Type &t = in.type;
    const std::string a = operand(in.args[0], t);
    const char *f = nullptr;
    switch (in.op) {
    case Op::Sqrt: f = "sqrtf"; break;
    case Op::RSqrt: f = "rsqrtf"; break;
    case Op::Exp: f = "expf"; break;
    case Op::Exp2: f = "exp2f"; break;
    case Op::Log: f = "logf"; break;
    case Op::Log2: f = "log2f"; break;
    case Op::Floor: f = "floorf"; break;
    case Op::Ceil: f = "ceilf"; break;
    case Op::Round: f = "roundf"; break;
    case Op::Trunc: f = "truncf"; break;
    case Op::Sin: f = "sinf"; break;
    case Op::Cos: f = "cosf"; break;
    case Op::Pow: f = "powf"; break;
    default: reject(in);
    }
    assign(in, std::string(f) + "(" + a + ")");
}

void Gen::emitSpecial(const Instr &in) {
    std::string expr;
    switch (in.op) {
    
    
    case Op::TidX: expr = "elemIndex"; break;
    case Op::CtaidX: expr = "(elemIndex / blockX)"; break;
    case Op::NtidX: expr = "blockX"; break;
    case Op::NtidY: expr = "blockY"; break;
    case Op::NtidZ: expr = "blockZ"; break;
    case Op::LaneId: expr = "(elemIndex % 32u)"; break;
    case Op::WarpId: expr = "(elemIndex / 32u)"; break;
    case Op::Popc: expr = "__builtin_popcount(elemIndex)"; break;
    case Op::NumWarps: expr = "((totalThreads + 31u) / 32u)"; break;
    default: reject(in);
    }
    assign(in, "static_cast<uint32_t>(" + expr + ")");
}

void Gen::emitCvt(const Instr &in) {
    if (in.args.size() != 1) reject(in);
    Type src = in.args[0].isValue() ? typeOf(in.args[0].value) : Type{};
    assign(in, std::string("(static_cast<") + cType(in.type) + ">(" +
                   operand(in.args[0], src) + "))");
}

void Gen::emitSelp(const Instr &in) {
    if (in.args.size() != 3) reject(in);
    Type t = in.args[0].isValue() ? typeOf(in.args[0].value)
                                  : Type{TypeKind::Int, 32, VecLen::Scalar};
    assign(in, "((" + operand(in.args[0], t) + ") ? (" + operand(in.args[1], in.type) +
                   ") : (" + operand(in.args[2], in.type) + "))");
}

void Gen::emitMove(const Instr &in) {
    if (in.args.size() != 1) reject(in);
    assign(in, "(" + operand(in.args[0], in.type) + ")");
}

Output Gen::run() {
    Output out;
    out.kernelName = ascend::sanitizeIdentifier(fn_.name);
    out.target = target_;
    out.params = fn_.params;

    for (size_t i = 0; i < fn_.params.size(); ++i)
        if (!fn_.params[i].name.empty())
            paramMap_[fn_.params[i].name] = paramName(i);

    
    for (const auto &b : fn_.blocks)
        for (const auto &in : b.instrs)
            if (in.hasResult() && !in.type.isVoid()) types_[in.result] = in.type;

    
    {
        std::vector<std::string> bad;
        for (const auto &b : fn_.blocks)
            for (const auto &in : b.instrs)
                if (!canLower(in.op)) {
                    bad.push_back(ir::opName(in.op));
                }
        if (!bad.empty()) {
            
            Op firstBad = Op::Unknown;
            std::ostringstream os;
            for (const auto &b : fn_.blocks)
                for (const auto &in : b.instrs)
                    if (!canLower(in.op) && firstBad == Op::Unknown) {
                        firstBad = in.op;
                        break;
                    }
            os << bad.size() << " unsupported operation(s): " << " (";
            for (size_t i = 0; i < bad.size(); ++i) {
                if (i) os << ", ";
                os << bad[i];
            }
            os << "). Reason: " << whyNot(firstBad);
            throw CodegenError(fn_.name, ir::opName(firstBad), os.str());
        }
    }

    
    for (const auto &b : fn_.blocks)
        for (const auto &in : b.instrs)
            if (in.hasResult() && !in.type.isVoid())
                decls_ << "        " << cType(in.type) << " " << valName(in.result)
                       << " = 0;\n";

    
    bool wrapNotPred = false;

    for (size_t bi = 0; bi < fn_.blocks.size(); ++bi) {
        const auto &blk = fn_.blocks[bi];
        const bool isFirst = (bi == 0);

        if (!isFirst) {
            const auto &prev = fn_.blocks[bi - 1];
            const Instr *last = prev.instrs.empty()
                                    ? nullptr
                                    : &prev.instrs[prev.instrs.size() - 1];
            if (!last || last->op != Op::CondBr) {
                if (last && last->op == Op::Ret) {
                    
                    continue;
                }
                wrapNotPred = false;
            } else if (!last->args.empty() &&
                       last->args[0].kind == Operand::Kind::LabelRef &&
                       last->args[0].block == static_cast<ir::BlockId>(bi)) {
                
                wrapNotPred = true;
            } else {
                
                line("if (" + valName(last->pred) + ") continue;");
                wrapNotPred = false;
            }
        }

        if (wrapNotPred) {
            
            const Instr *last = &fn_.blocks[bi - 1].instrs.back();
            line("if (!" + valName(last->pred) + ") {");
            ++indent_;
        }

        for (const auto &in : blk.instrs) {
            switch (in.op) {
            case Op::Ret: continue;
            case Op::Br:
                throw CodegenError(fn_.name, "br",
                                   "an unconditional branch is not represented "
                                   "in the unrolled model: a loop needs a "
                                   "structural transformation");
            case Op::CondBr:
                
                continue;
            case Op::Load: emitLoad(in); break;
            case Op::Store: emitStore(in); break;
            case Op::Add: case Op::Sub: case Op::Mul:
            case Op::DivS: case Op::DivU: case Op::RemS: case Op::RemU:
            case Op::And: case Op::Or: case Op::Xor:
            case Op::Shl: case Op::ShrU:
                emitArith(in);
                break;
            case Op::FAdd: case Op::FSub: case Op::FMul: case Op::FDiv:
                emitFloat(in);
                break;
            case Op::FFma:
                if (in.args.size() != 3) reject(in);
                assign(in, "(fmaf(" + operand(in.args[0], in.type) + ", " +
                               operand(in.args[1], in.type) + ", " +
                               operand(in.args[2], in.type) + "))");
                break;
            case Op::Neg: case Op::FNeg: case Op::Not: case Op::FAbs:
                emitUnary(in);
                break;
            case Op::CmpEq: case Op::CmpNe:
            case Op::CmpLtS: case Op::CmpLtU: case Op::CmpLeS: case Op::CmpLeU:
            case Op::CmpGtS: case Op::CmpGtU: case Op::CmpGeS: case Op::CmpGeU:
                emitIntCmp(in);
                break;
            case Op::FCmpEq: case Op::FCmpNe: case Op::FCmpLt:
            case Op::FCmpLe: case Op::FCmpGt: case Op::FCmpGe:
                emitFloatCmp(in);
                break;
            case Op::Sqrt: case Op::RSqrt: case Op::Exp: case Op::Exp2:
            case Op::Log: case Op::Log2: case Op::Floor: case Op::Ceil:
            case Op::Round: case Op::Trunc: case Op::Sin: case Op::Cos:
            case Op::Pow:
                emitMath(in);
                break;
            case Op::Cvt: emitCvt(in); break;
            case Op::Selp: emitSelp(in); break;
            case Op::Mov: emitMove(in); break;
            case Op::BarrierThreads:
                
                line("AscendC::SyncAll();");
                break;
            case Op::TidX: case Op::CtaidX: case Op::LaneId:
            case Op::WarpId: case Op::Popc: case Op::NumWarps:
                emitSpecial(in);
                break;
            default:
                reject(in);
            }
        }

        if (wrapNotPred) {
            --indent_;
            line("}");
        }
    }

    
    std::ostringstream params;
    for (size_t i = 0; i < fn_.params.size(); ++i) {
        if (i) params << ", ";
        const auto &p = fn_.params[i];
        const bool byPtr = p.isPointer || (p.type.isInt() && p.type.width == 64);
        if (byPtr) params << "GM_ADDR " << paramName(i);
        else params << cType(p.type) << " " << paramName(i);
    }

    std::ostringstream src;
    src << "// Generated by cuascend. Do not edit.\n"
        << "// Kernel:  " << fn_.name << "\n"
        << "// Target:  " << target_.socVersion << "\n"
        << "//\n"
        << "// SIMT is unrolled into a linear space: a CUDA thread becomes\n"
        << "// the element index elemIndex, and the range is split across cores.\n"
        << "\n"
        << "#include \"kernel_operator.h\"\n"
        << "\n"
        << "#include \"" << ascend::sanitizeIdentifier(fn_.name) << "_tiling_data.h\"\n"
        << "using namespace AscendC;\n"
        << "\n"
        
        << "extern \"C\" __global__ __aicore__ void " << ascend::sanitizeIdentifier(fn_.name)
        << "(GM_ADDR workspace, GM_ADDR tiling)\n"
        << "{\n"
        << "    (void)workspace;\n"
        << "    GET_TILING_DATA(tiling_data, tiling);\n"
        << "    const uint32_t linearThreads = tiling_data.linearThreads;\n"
        << "    const uint32_t blockX = tiling_data.blockX;\n"
        << "    const uint32_t blockY = tiling_data.blockY;\n"
        << "    const uint32_t blockZ = tiling_data.blockZ;\n"
        << "    const uint32_t gridX = tiling_data.gridX;\n"
        << "\n"
        << "    const uint32_t coreNum = AscendC::GetBlockNum();\n"
        << "    const uint32_t coreIdx = AscendC::GetBlockIdx();\n"
        << "    const uint32_t totalThreads = linearThreads;\n"
        << "\n"
        << "    const uint32_t perCore = (totalThreads + coreNum - 1u) / coreNum;\n"
        << "    const uint32_t rangeStart = coreIdx * perCore;\n"
        << "    uint32_t rangeEnd = rangeStart + perCore;\n"
        << "    if (rangeEnd > totalThreads) rangeEnd = totalThreads;\n"
        << "\n"
        << hoist_.str()
        << "    for (uint32_t elemIndex = rangeStart; elemIndex < rangeEnd; ++elemIndex) {\n"
        << decls_.str()
        << body_.str()
        << "    }\n"
        << "}\n";

    out.source = src.str();
    return out;
}

} 

std::vector<std::string> unsupported(const ir::Function &fn) {
    std::vector<std::string> out;
    for (const auto &b : fn.blocks)
        for (const auto &in : b.instrs)
            if (!canLower(in.op)) {
                std::string msg = ir::opName(in.op);
                msg += ": ";
                msg += whyNot(in.op);
                out.push_back(std::move(msg));
            }
    return out;
}

Output generate(const ir::Function &fn, const TargetInfo &target) {
    return Gen(fn, target).run();
}

} 

// MAIN: the compile pipeline

#include <algorithm>
#include <cctype>
#include <array>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace ca::ascend {
namespace fs = std::filesystem;

namespace {

std::string g_lastError;
} 

std::string ascendLastError() { return g_lastError; }

namespace {

std::string envOr(const char *name, const std::string &fallback) {
    const char *v = std::getenv(name);
    return (v && *v) ? std::string(v) : fallback;
}

bool isExecutable(const std::string &p) {
    if (p.empty()) return false;
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) return false;
    const auto perms = fs::status(p, ec).permissions();
    return (perms & fs::perms::owner_exec) != fs::perms::none;
}

} 

Toolchain Toolchain::fromEnvironment() {
    Toolchain tc;
    tc.root = envOr("CUASCEND_CANN_ROOT",
                    envOr("ASCEND_HOME_PATH",
                          "/usr/local/Ascend/ascend-toolkit/latest"));
    tc.bisheng = envOr("CUASCEND_BISHENG", tc.root + "/compiler/ccec_compiler/bin/bisheng");
    
    tc.msopgen = envOr("CUASCEND_MSOPGEN", "");
    if (tc.msopgen.empty()) {
        for (const char *cand : {"/aarch64-linux/bin/msopgen", "/bin/msopgen",
                                 "/tools/msopgen/msopgen.py"}) {
            const std::string p = tc.root + cand;
            if (std::filesystem::exists(p)) {
                tc.msopgen = p;
                break;
            }
        }
        if (tc.msopgen.empty())
            tc.msopgen = tc.root + "/aarch64-linux/bin/msopgen";
    }
    tc.socVersion = envOr("CUASCEND_SOC", "Ascend910B1");
    tc.workDir = envOr("CUASCEND_WORKDIR", "/tmp/cuascend-kernels");
    tc.verbose = envOr("CUASCEND_VERBOSE", "") == "1";
    return tc;
}

std::vector<std::string> ascendcIncludeDirs(const Toolchain &tc) {
    const char *explicitList = std::getenv("CUASCEND_ASCENDC_INCLUDE");
    std::vector<std::string> dirs;
    if (explicitList && *explicitList) {
        std::string cur;
        for (const char *p = explicitList;; ++p) {
            if (*p == ':' || *p == '\0') {
                if (!cur.empty()) dirs.push_back(cur);
                cur.clear();
                if (*p == '\0') break;
            } else {
                cur.push_back(*p);
            }
        }
        return dirs;
    }

    
    const fs::path roots[] = {
        fs::path(tc.root) / "aarch64-linux" / "ascendc" / "include",
        fs::path(tc.root) / "include" / "ascendc",
        fs::path(tc.root) / "tikcpp",
    };
    std::error_code ec;

    
    auto addParent = [&dirs](const fs::path &p) {
        const fs::path parent = p.parent_path();
        if (parent.empty()) return;
        const std::string s = parent.string();
        if (std::find(dirs.begin(), dirs.end(), s) == dirs.end())
            dirs.push_back(s);
    };

    for (const auto &root : roots) {
        if (!fs::is_directory(root, ec)) continue;
        dirs.push_back(root.string());
        addParent(root);
        for (auto it = fs::recursive_directory_iterator(
                 root,
                 fs::directory_options::skip_permission_denied |
                     fs::directory_options::follow_directory_symlink,
                 ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            if (!it->is_directory(ec)) continue;
            const fs::path p = it->path();
            if (p.string().size() > root.string().size() + 40) {
                it.disable_recursion_pending();
                continue;
            }
            dirs.push_back(p.string());
            addParent(p);
        }
    }
    return dirs;
}

std::string hashSource(const std::string &s) {
    
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

std::string runCommand(const std::vector<std::string> &argv, bool verbose) {
    if (argv.empty())
        throw CompileError("empty command", "");

    if (verbose) {
        std::string line = "[cuascend] run:";
        for (const auto &a : argv) line += " " + a;
        std::fprintf(stderr, "%s\n", line.c_str());
    }

    int pipefd[2];
    if (::pipe(pipefd) != 0)
        throw CompileError("cannot create pipe for: " + argv[0], "");

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        throw CompileError("fork failed for: " + argv[0], "");
    }

    if (pid == 0) {
        ::close(pipefd[0]);
        
        ::unsetenv("LD_PRELOAD");
        ::dup2(pipefd[1], STDOUT_FILENO);
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);
        
        std::vector<char *> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto &a : argv)
            cargv.push_back(const_cast<char *>(a.c_str()));
        cargv.push_back(nullptr);
        ::execvp(cargv[0], cargv.data());
        
        const std::string m = std::string("exec failed: ") + cargv[0] + "\n";
        ssize_t ignored = ::write(pipefd[1], m.data(), m.size());
        (void)ignored;
        ::_exit(127);
    }

    ::close(pipefd[1]);
    std::string output;
    char buf[4096];
    ssize_t n;
    while ((n = ::read(pipefd[0], buf, sizeof buf)) > 0)
        output.append(buf, static_cast<size_t>(n));
    ::close(pipefd[0]);

    int status = 0;
    ::waitpid(pid, &status, 0);

    if (WIFSIGNALED(status))
        throw CompileError(argv[0] + " killed by signal " +
                               std::to_string(WTERMSIG(status)),
                           output);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw CompileError(argv[0] + " exited with status " +
                               std::to_string(WEXITSTATUS(status)),
                           output);

    return output;
}

namespace {

ElemType elemTypeOf(const ir::Type &t) {
    if (t.isFloat()) {
        if (t.width == 32) return ElemType::Float32;
        if (t.width == 16) return ElemType::Float16;
        return ElemType::Float32;
    }
    if (t.isBool()) return ElemType::Uint8;
    switch (t.width) {
    case 8: return t.isSigned() ? ElemType::Int8 : ElemType::Uint8;
    case 16: return t.isSigned() ? ElemType::Int16 : ElemType::Uint16;
    case 32: return t.isSigned() ? ElemType::Int32 : ElemType::Uint32;
    case 64: return t.isSigned() ? ElemType::Int64 : ElemType::Uint64;
    default: break;
    }
    return ElemType::Unknown;
}

uint32_t elemSizeOf(const ir::Type &t) {
    if (t.isBool()) return 1;
    const uint32_t bits = t.width ? t.width : 32;
    return (bits + 7) / 8;
}

} 

fs::path ensureCannDevLayout(const Toolchain &tc) {
    std::error_code ec;

    
    fs::path arch;
    {
        std::vector<fs::path> roots{fs::path(tc.root)};
        for (const auto &e : fs::directory_iterator(fs::path(tc.root), ec))
            roots.push_back(e.path());
        for (const auto &r : roots) {
            if (fs::exists(r / "include/register/tilingdata_base.h", ec)) {
                arch = r;
                break;
            }
        }
    }
    if (arch.empty())
        return {};

    
    fs::path version;
    for (fs::path d = arch; !d.empty() && d != "/"; d = d.parent_path()) {
        if (fs::exists(d / "tools/opbuild/op_build", ec)) {
            version = d;
            break;
        }
    }
    if (version.empty())
        version = arch;

    
    fs::path compiler;
    for (fs::path d = arch; !d.empty() && d != "/"; d = d.parent_path()) {
        if (fs::exists(d / "compiler/ccec_compiler/bin/bisheng", ec)) {
            compiler = d / "compiler";
            break;
        }
    }

    const fs::path dev = "/usr/local/Ascend/canndev";
    fs::create_directories(dev, ec);
    auto link = [&](const char *name, const fs::path &target) {
        if (target.empty()) return;
        fs::remove(dev / name, ec);
        fs::create_symlink(target, dev / name, ec);
    };
    link("toolkit", version);
    if (!compiler.empty())
        link("compiler", compiler);
    link("include", arch / "include");

    
    fs::remove(dev / "lib64", ec);
    fs::create_directories(dev / "lib64", ec);
    std::error_code walk;
    
    for (const auto &e : fs::directory_iterator(arch / "lib64", walk)) {
        if (!e.is_regular_file()) continue;
        const fs::path target = fs::canonical(e.path(), ec);
        if (target.empty()) continue;
        fs::create_symlink(target, dev / "lib64" / e.path().filename(), ec);
    }
    fs::create_symlink(arch / "lib64" / "libtiling.so",
                       dev / "lib64" / "libtiling_api.so", ec);
    return dev;
}

bool isTensorParam(const ir::Param &p) {
    return p.isPointer || (p.type.isInt() && p.type.width == 64);
}

void writeTilingHeader(const fs::path &outDir, const std::string &kernelName,
                       const ir::Function &fn) {
    std::error_code ec;
    fs::create_directories(outDir, ec);

    std::string structName = kernelName;
    structName[0] = static_cast<char>(
        std::toupper(static_cast<unsigned char>(structName[0])));
    structName += "TilingData";

    const std::string guard = "__" + structName + "_H__";
    std::ostringstream os;
    os << "#ifndef " << guard << "\n";
    os << "#define " << guard << "\n\n";
    os << "#include <cstdint>\n";
    os << "#include <cstring>\n\n";
    os << "#include \"kernel_operator.h\"\n";
    os << "using namespace AscendC;\n\n";
    os << "#include <cstring>\n\n";
    os << "#pragma pack(1)\n";
    os << "struct " << structName << " {\n";
    for (size_t i = 0; i < fn.params.size(); ++i) {
        const ir::Param &p = fn.params[i];
        const bool isPointer = isTensorParam(p);
        std::string ctype;
        if (isPointer) ctype = "uint64_t";
        else if (p.type.isFloat()) ctype = "float";
        else ctype = "uint32_t";
        os << "    " << ctype << " arg" << i << ";\n";
    }
    os << "    uint32_t linearThreads;\n";
    os << "    uint32_t blockX;\n";
    os << "    uint32_t blockY;\n";
    os << "    uint32_t blockZ;\n";
    os << "    uint32_t gridX;\n";
    os << "    uint32_t size;\n";
    os << "};\n";
    os << "#pragma pack()\n\n";
    
    os << "inline [aicore] void Init" << structName
       << "(GM_ADDR tiling, " << structName << "* const_data)\n";
    os << "{\n";
    os << "    const __gm__ uint64_t *src = (const __gm__ uint64_t *)tiling;\n";
    os << "    uint64_t *dst = (uint64_t *)const_data;\n";
    os << "    for (auto i = 0; i < sizeof(" << structName
       << ") / 8; i++) *(dst + i) = *(src + i);\n";
    os << "}\n\n";
    
    os << "#define GET_TILING_DATA(tiling_data, tiling_arg) \\\n";
    os << structName << " tiling_data; \\\n";
    os << "Init" << structName
       << "(tiling_arg, &tiling_data)\n";
    os << "#endif\n";

    std::ofstream fo(outDir / (kernelName + "_tiling_data.h"));
    if (!fo)
        throw CompileError("cannot write tiling data header for " + kernelName, "");
    fo << os.str();
}

void annotateOpModel(const fs::path &path, const ir::Function &fn) {
    std::ifstream in(path);
    if (!in) return;
    const std::string src((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

    
    const std::string op = canonicalOpName(fn.name);
    const size_t opAt = src.find('"' + op + '"');
    if (opAt == std::string::npos) return;
    
    std::vector<size_t> ptrs;
    for (size_t i = 0; i < fn.params.size(); ++i) {
        const ir::Param &p = fn.params[i];
        if (p.isPointer || (p.type.isInt() && p.type.width == 64))
            ptrs.push_back(i);
    }
    if (ptrs.empty()) return;
    const size_t outIdx = ptrs.back();

    std::ostringstream os;
    os << "{\n    \"name\": \"" << sanitizeIdentifier(fn.params[outIdx].name)
       << "\",\n"
       << "    \"paramType\": \"required\",\n"
       << "    \"dtype\": \"" << cannDtype(fn.params[outIdx].type) << "\",\n"
       << "    \"format\": \"ND\",\n"
       << "    \"shape\": \"all\",\n"
       << "    \"needCompile\": \"false\"\n  }";
    const std::string outDesc = os.str();

    std::string extra;
    size_t tensorNo = 0;
    for (size_t idx : ptrs) {
        std::ostringstream t;
        t << "  \"input" << tensorNo << "\": {\n"
          << "    \"name\": \"" << sanitizeIdentifier(fn.params[idx].name)
          << "\",\n"
          << "    \"paramType\": \"required\",\n"
          << "    \"dtype\": \"" << cannDtype(fn.params[idx].type) << "\",\n"
          << "    \"format\": \"ND\",\n"
          << "    \"shape\": \"all\",\n"
          << "    \"needCompile\": \"false\"\n"
          << "  },\n";
        extra += t.str();
        ++tensorNo;
    }
    extra += "  \"output0\": " + outDesc + ",\n";
    extra += "  \"compute\": {\"cost\": \"10\"},\n";

    
    const size_t openBrace = src.find('{', opAt);
    if (openBrace == std::string::npos) return;
    std::string out = src.substr(0, openBrace + 1);
    out += "\n";
    out += extra;
    out += src.substr(openBrace + 1);

    
    {
        const std::string key = "\"coreType\"";
        size_t k = out.find(key);
        while (k != std::string::npos) {
            const size_t colon = out.find(':', k);
            size_t b = out.find('{', colon);
            int depth = 0;
            size_t e = b;
            for (; e < out.size(); ++e) {
                if (out[e] == '{') ++depth;
                else if (out[e] == '}' && --depth == 0) { ++e; break; }
            }
            if (out[e] == ',') ++e;
            out.erase(k, e - k);
            k = out.find(key, k);
        }
    }

    std::ofstream fo(path);
    if (fo) fo << out;
}

std::string runCommandIn(const fs::path &cwd,
                         const std::vector<std::string> &argv, bool verbose) {
    if (argv.empty())
        throw CompileError("empty command", "");

    
    setenv("CUASCEND_BUILD_CWD", cwd.c_str(), 1);

    const int saved = ::open(".", O_RDONLY | O_DIRECTORY);
    if (saved < 0)
        throw CompileError("cannot open cwd for build.sh", "");
    if (::chdir(cwd.c_str()) != 0) {
        ::close(saved);
        throw CompileError("cannot enter " + cwd.string(), "");
    }
    std::string out;
    try {
        out = runCommand(argv, verbose);
    } catch (...) {
        if (::fchdir(saved) != 0) {  }
        ::close(saved);
        throw;
    }
    if (::fchdir(saved) != 0) {  }
    ::close(saved);
    return out;
}

CompiledOp compileKernel(const std::string &ptxSource,
                         const std::string &kernelName,
                         const std::string &socVersion,
                         const Toolchain &tc) {
    
    ptx::Module pm;
    try {
        pm = ptx::parse(ptxSource, "<fatbin>");
    } catch (const ptx::Error &e) {
        throw CompileError(std::string("PTX parse failed: ") + e.what(), "");
    }

    const ptx::Function *entry = pm.findEntry(kernelName);
    if (!entry)
        throw CompileError("no .entry named '" + kernelName + "' in module", "");

    lower::Options lopt;
    const lower::Result lowered = lower::run(pm, lopt);

    const ir::Function *f = lowered.module.find(kernelName);
    if (!f) throw CompileError("kernel '" + kernelName + "' not produced by lowering", "");

    const auto problems = ir::verify(lowered.module);
    if (!problems.empty()) {
        std::ostringstream os;
        os << "IR verification failed for '" << kernelName << "':";
        for (const auto &p : problems) os << "\n  " << p;
        throw CompileError(os.str(), "");
    }

    
    codegen::TargetInfo target;
    target.socVersion = socVersion;
    codegen::Output gen;
    try {
        gen = codegen::generate(*f, target);
    } catch (const codegen::CodegenError &e) {
        throw CompileError(e.what(), "");
    }

    
    const std::string hash = hashSource(ptxSource + "@" + socVersion);
    const fs::path dir = fs::path(tc.workDir) / hash;
    // The file names come from the name in the image, so they are made safe before they become paths.
    const std::string fileStem = sanitizeIdentifier(kernelName);
    std::error_code ec;
    fs::create_directories(dir, ec);

    const fs::path src = dir / (fileStem + ".cpp");
    {
        std::ofstream f2(src);
        if (!f2) throw CompileError("cannot write " + src.string(), "");
        f2 << gen.source;

    
    writeTilingHeader(dir, fileStem, *f);
    }

    const fs::path obj = dir / (fileStem + ".o");

    
    const std::string archFlags = envOr(
        "CUASCEND_BISHENG_FLAGS",
        std::string("--cce-aicore-arch=") +
            envOr("CUASCEND_AICORE_ARCH", "dav-c220") +
            " --cce-aicore-lang -std=c++17 -O2 -fPIC -c");
    
    if (!isExecutable(tc.bisheng)) {
        throw CompileError(
            "bisheng not found at: " + tc.bisheng +
                ". Set CUASCEND_BISHENG to the compiler path; with CANN present "
                "it is $ASCEND_HOME_PATH/compiler/ccec_compiler/bin/bisheng",
            "");
    }

    std::vector<std::string> bishengCmd;
    bishengCmd.insert(bishengCmd.end(), {"-I", dir.string()});
    
    for (const auto &d : ascendcIncludeDirs(tc))
        bishengCmd.insert(bishengCmd.end(), {"-I", d});
    {
        std::istringstream is(archFlags);
        std::string tok;
        while (is >> tok) bishengCmd.push_back(tok);
    }
    bishengCmd.push_back("-o");
    bishengCmd.push_back(obj.string());
    bishengCmd.push_back(src.string());

    std::string bishengOut;
    try {
        
        bishengCmd[0] = tc.bisheng;
        bishengOut = runCommand(bishengCmd, tc.verbose);
    } catch (const CompileError &e) {
        throw CompileError("bisheng failed for kernel '" + kernelName + "'",
                           e.output);
    }

    if (!fs::exists(obj)) {
        std::ostringstream os;
        os << "bisheng reported success but produced no object file at "
           << obj.string();
        if (!bishengOut.empty()) os << "\ncompiler output:\n" << bishengOut;
        throw CompileError(os.str(), bishengOut);
    }

    
    const std::string opDesc = envOr("CUASCEND_OP_INFO", "");
    fs::path descPath;
    if (!opDesc.empty()) {
        descPath = opDesc;
    } else {
        descPath = dir / (kernelName + "_op_info.json");
        
        size_t nOut = 0;
        for (size_t i = f->params.size(); i-- > 0;) {
            const ir::Param &p = f->params[i];
            if (p.isPointer || (p.type.isInt() && p.type.width == 64)) {
                nOut = 1;
                break;
            }
        }
        const std::string json = generateOpInfo(*f, 0, nOut);
        std::ofstream fo(descPath);
        if (!fo)
            throw CompileError("cannot write " + descPath.string(), "");
        fo << json;
        if (!fo)
            throw CompileError("cannot write " + descPath.string(), "");
        if (tc.verbose)
            std::fprintf(stderr, "[cuascend] op_info for %s:\n%s", kernelName.c_str(),
                         json.c_str());
    }

    
    
    const fs::path outDir = dir / "op";

    
    fs::remove_all(outDir, ec);
    fs::create_directories(outDir, ec);

    
    const int jobs = std::atoi(std::getenv("CUASCEND_BUILD_JOBS")
                                   ? std::getenv("CUASCEND_BUILD_JOBS")
                                   : "4");

    
    std::string cu = socVersion;
    std::transform(cu.begin(), cu.end(), cu.begin(),
                   [](unsigned char c) { return static_cast<char>(::tolower(c)); });
    const std::string unit = "ai_core-" + cu;

    if (!isExecutable(tc.msopgen)) {
        throw CompileError(
            "msopgen not found at: " + tc.msopgen +
                ". Set CUASCEND_MSOPGEN to the generator path; with CANN "
                "present it is $ASCEND_HOME_PATH/aarch64-linux/bin/msopgen",
            "");
    }

    std::vector<std::string> msopCmd{tc.msopgen, "gen",
                                     "-i", descPath.string(),
                                     "-c", unit,
                                     "-lan", "cpp",
                                     "-out", outDir.string()};
    if (tc.verbose) {
        std::string line = "[cuascend] msopgen:";
        for (const auto &a : msopCmd) line += " " + a;
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    std::string msopOut;
    try {
        msopOut = runCommand(msopCmd, tc.verbose);
        bishengOut += msopOut;
    } catch (const CompileError &e) {
        throw CompileError("msopgen failed for kernel '" + kernelName + "'",
                           e.output);
    }

    
    const fs::path projBuild = outDir / "build_out";
    {
        
        const fs::path cannDev = ensureCannDevLayout(tc);
        const char *cannEnv = std::getenv("CUASCEND_CANN_PACKAGE_PATH");
        const std::string cannPath =
            cannEnv ? cannEnv : cannDev.string();

        
        {
            const fs::path presets = outDir / "CMakePresets.json";
            std::ifstream pin(presets);
            if (pin && !cannPath.empty()) {
                const std::string text(
                    (std::istreambuf_iterator<char>(pin)),
                    std::istreambuf_iterator<char>());
                const std::string key = "\"ASCEND_CANN_PACKAGE_PATH\"";
                const size_t k = text.find(key);
                if (k != std::string::npos) {
                    const std::string val = "\"value\": \"";
                    const size_t v = text.find(val, k);
                    if (v != std::string::npos) {
                        const size_t start = v + val.size();
                        const size_t e = text.find('"', start);
                        std::string patched = text.substr(0, start);
                        patched += cannPath;
                        patched += text.substr(e);
                        std::ofstream pout(presets);
                        if (pout) pout << patched;
                    }
                }
            }
        }

        
        const std::string inner =
            "ASCEND_CANN_PACKAGE_PATH=" + cannPath + " ./build.sh";
        std::vector<std::string> buildCmd{"bash", "-c", inner};

        
        try {
            bishengOut += runCommandIn(outDir, buildCmd, tc.verbose);
        } catch (const CompileError &e) {
            std::ostringstream os;
            os << "building the generated op project failed. The kernel "
                  "subproject is produced by build.sh; without it the "
                  "package carries no machine code and the op cannot run.";
            g_lastError = os.str();
            throw CompileError(os.str(), e.output);
        }
    }
        (void)jobs;

    
    {
        std::vector<std::string> pkg{"cmake", "--build", projBuild.string(),
                                     "--target", "package"};
        try {
            bishengOut += runCommand(pkg, tc.verbose);
        } catch (const CompileError &e) {
            std::ostringstream os;
            os << "packaging the generated op project failed. The runtime "
                  "needs vendors/customize with the kernel and the op "
                  "libraries; without it the op loads but cannot run.";
            g_lastError = os.str();
            throw CompileError(os.str(), e.output);
        }
    }

    
    {
        std::error_code me;
        for (const auto &e : fs::recursive_directory_iterator(
                 outDir / "build_out", fs::directory_options::skip_permission_denied, me)) {
            if (!e.is_regular_file()) continue;
            const std::string nm = e.path().filename().string();
            if (nm.find("-ops-info.json") == std::string::npos) continue;
            annotateOpModel(e.path(), *f);
        }
    }

    
    {
        fs::path pkg;
        std::error_code pe;
        for (const auto &e : fs::directory_iterator(outDir / "build_out", pe)) {
            if (e.is_regular_file() && e.path().extension() == ".run") {
                pkg = e.path();
                break;
            }
        }
        const char *oppRoot =
            std::getenv("CUASCEND_OPP_ROOT");
        const fs::path opp = oppRoot ? fs::path(oppRoot)
                                     : fs::path(tc.root) / "opp";
        if (!pkg.empty()) {
            std::vector<std::string> inst{"bash", pkg.string(),
                                          "--install-path=" + opp.string(),
                                          "--install-for-all", "--nox11"};
            try {
                bishengOut += runCommand(inst, tc.verbose);
            } catch (const CompileError &e2) {
                std::ostringstream os;
                os << "installing the custom op package into "
                      << opp.string() << " failed";
                g_lastError = os.str();
                throw CompileError(os.str(), e2.output);
            }
        }
    }

    
    fs::path jsonPath;
    fs::path binPath;
    
    for (const char *sub : {"build_out", "build"}) {
        if (!jsonPath.empty()) break;
        std::error_code we;
        const fs::path rootDir = outDir / sub;
        if (!fs::is_directory(rootDir, we)) continue;
        for (const auto &entry : fs::recursive_directory_iterator(
                 rootDir, fs::directory_options::skip_permission_denied, we)) {
            if (!entry.is_regular_file()) continue;
            const fs::path &p = entry.path();
            const std::string name = p.filename().string();
            
            if (p.extension() == ".json" &&
                name.find("-ops-info.json") != std::string::npos &&
                jsonPath.empty())
                jsonPath = p;
            if (p.extension() == ".bin" && binPath.empty())
                binPath = p;
        }
    }
    if (jsonPath.empty())
        throw CompileError("msopgen produced no op model for '" + kernelName +
                               "'; aclopLoadOpWithFile needs a .json descriptor",
                           bishengOut);

    CompiledOp op;
    op.valid = true;
    op.ir = std::make_shared<ir::Function>(*f);
    op.kernelName = canonicalOpName(kernelName);
    op.modelPath = jsonPath.string();
    op.binPath = binPath.empty() ? std::string() : binPath.string();
    op.sourceDir = dir.string();
    op.sourceHash = hash;
    op.inputCount = f->params.size();
    
    op.outputCount = 0;
    for (size_t i = f->params.size(); i-- > 0;) {
        const ir::Param &p = f->params[i];
        if (p.isPointer || (p.type.isInt() && p.type.width == 64)) {
            op.outputCount = 1;
            break;
        }
    }

    
    for (const auto &p : f->params) {
        op.paramTypes.push_back(elemTypeOf(p.type));
        op.paramSizes.push_back(elemSizeOf(p.type));
    }
    return op;
}

size_t countModelOutputs(const fs::path &modelPath) {
    if (modelPath.empty()) return 0;
    std::ifstream in(modelPath);
    if (!in) return 0;
    size_t count = 0;
    std::string line;
    while (std::getline(in, line)) {
        size_t at = 0;
        while ((at = line.find("\"output", at)) != std::string::npos) {
            at += 7;
            if (at < line.size() && std::isdigit(static_cast<unsigned char>(
                                        line[at])))
                ++count;
        }
    }
    return count;
}

size_t countModelInputs(const fs::path &modelPath) {
    if (modelPath.empty()) return 0;
    std::ifstream in(modelPath);
    if (!in) return 0;
    size_t count = 0;
    std::string line;
    while (std::getline(in, line)) {
        size_t at = 0;
        while ((at = line.find("\"input", at)) != std::string::npos) {
            at += 6;
            if (at < line.size() && std::isdigit(static_cast<unsigned char>(
                                        line[at])))
                ++count;
        }
    }
    return count;
}

CompiledOp OpCache::get(const std::string &ptxSource,
                        const std::string &kernelName, const Toolchain &tc) {
    
    const std::string hash =
        hashSource(ptxSource + "@" + kernelName + "@" + tc.socVersion);

    for (const auto &kv : entries_) {
        if (kv.first == hash) {
            ++hits_;
            if (kv.second.valid) return kv.second;
        }
    }

    
    const fs::path dir = fs::path(tc.workDir) / hash;
    const fs::path marker = dir / "ok";
    std::error_code ec;
    if (fs::exists(marker, ec)) {
        CompiledOp op;
        op.valid = true;
        op.kernelName = canonicalOpName(kernelName);
        op.sourceDir = dir.string();
        op.sourceHash = hash;
        
        op.inputCount = 0;
        
        for (const auto &e : fs::recursive_directory_iterator(
                 dir / "op", fs::directory_options::skip_permission_denied, ec)) {
            if (!e.is_regular_file()) continue;
            const fs::path &p = e.path();
            const std::string name = p.filename().string();
            if (p.extension() == ".json" &&
                name.find("-ops-info.json") != std::string::npos &&
                op.modelPath.empty())
                op.modelPath = p.string();
            if (p.extension() == ".o" && op.binPath.empty())
                op.binPath = p.string();
        }
        
        {
            try {
                ptx::Module pm;
                try {
                    pm = ptx::parse(ptxSource, "<fatbin>");
                } catch (const ptx::Error &e) {
                    throw CompileError(std::string("PTX parse failed: ") + e.what(),
                                      "");
                }
                ir::Module im = lower::run(pm).module;
                for (const auto &f : im.functions) {
                    if (f.name == kernelName || im.functions.size() == 1) {
                        op.ir = std::make_shared<ir::Function>(f);
                        break;
                    }
                }
            } catch (...) {
                
            }
        }
        op.inputCount = countModelInputs(op.modelPath);
        op.outputCount = countModelOutputs(op.modelPath);

        if (op.modelPath.empty()) {
            
            fs::remove(marker, ec);
        } else {
            ++hits_;
            entries_.emplace_back(hash, op);
            return op;
        }
    }

    ++misses_;
    CompiledOp op = compileKernel(ptxSource, kernelName, tc.socVersion, tc);

    std::ofstream m(marker);
    if (m) m << op.modelPath << "\n" << op.binPath << "\n";

    entries_.emplace_back(hash, op);
    return op;
}

namespace {

constexpr int kFormatND = -1;

std::string quote(const std::string &s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    out += '"';
    return out;
}

std::string shapeList(const std::vector<int64_t> &dims) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) os << ", ";
        os << dims[i];
    }
    os << ']';
    return os.str();
}

std::string tensorJson(const TensorDesc &t, const char *indent) {
    
    
    
    
    std::ostringstream os;
    os << indent << "{\n";
    os << indent << "  \"name\": " << quote(t.name.empty() ? "tensor" : t.name) << ",\n";
    os << indent << "  \"type\": [" << quote(t.dtype) << "],\n";
    os << indent << "  \"format\": [" << quote("ND") << "],\n";
    os << indent << "  \"ori_format\": [" << quote("ND") << "],\n";
    os << indent << "  \"shape\": " << shapeList(t.shape) << ",\n";
    os << indent << "  \"ori_shape\": " << shapeList(t.shape) << ",\n";
    os << indent << "  \"paramType\": "
       << quote(t.paramType.empty() ? "required" : t.paramType) << ",\n";
    os << indent << "  \"value\": " << quote(t.value) << "\n";
    os << indent << '}';
    return os.str();
}

std::string join(const std::vector<TensorDesc> &tensors, const char *indent) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i) os << ",\n";
        os << tensorJson(tensors[i], indent);
    }
    os << "]";
    return os.str();
}

} 

// An identifier that is safe to put into generated sources and into file names: the names come from the PTX image, which is data, and the package is compiled on the host.
std::string sanitizeIdentifier(const std::string &name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        const bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        const bool digit = c >= '0' && c <= '9';
        out += (letter || digit || c == '_') ? c : '_';
    }
    if (out.empty()) out = "_";
    if (out[0] >= '0' && out[0] <= '9') out.insert(out.begin(), '_');
    return out;
}

std::string canonicalOpName(const std::string &fnName) {
    
    
    
    std::string out;
    bool upper = true;
    for (char c : fnName) {
        if (c == '_' || c == '.' || c == ':') { upper = true; continue; }
        out += upper ? static_cast<char>(std::toupper(static_cast<unsigned char>(c)))
                     : c;
        upper = false;
    }
    while (!out.empty() && !std::isupper(static_cast<unsigned char>(out[0])))
        out.erase(out.begin());
    return out.empty() ? std::string("CustomOp") : out;
}

std::string cannDtype(const ir::Type &t) {
    if (t.isFloat()) return t.width == 16 ? "float16" : "float32";
    if (t.isBool()) return "int8";
    switch (t.width) {
    case 8: return t.isSigned() ? "int8" : "uint8";
    case 16: return t.isSigned() ? "int16" : "uint16";
    case 32: return t.isSigned() ? "int32" : "uint32";
    case 64: return t.isSigned() ? "int64" : "uint64";
    default: break;
    }
    return "float32";
}

std::string generateOpInfo(const ir::Function &fn, size_t inputCount,
                           size_t outputCount, int format) {
    
    
    
    
    std::vector<TensorDesc> in, out, attrs;
    for (size_t i = 0; i < fn.params.size(); ++i) {
        const ir::Param &p = fn.params[i];
        const bool isPointer = p.isPointer || (p.type.isInt() && p.type.width == 64);
        if (!isPointer) {
            TensorDesc a;
            a.shape = {1};
            a.format = kFormatND;
            a.dtype = cannDtype(p.type);
            a.value = "0";
            attrs.push_back(std::move(a));
            continue;
        }
        TensorDesc t;
        t.name = sanitizeIdentifier(p.name);
        t.paramType = "required";
        t.shape = {-1};
        t.format = format >= 0 ? format : kFormatND;
        t.dtype = cannDtype(p.type);
        const size_t fromEnd = fn.params.size() - i;
        if (outputCount > 0 && fromEnd <= outputCount) out.push_back(std::move(t));
        else in.push_back(std::move(t));
    }
    (void)inputCount;
    return generateOpInfo(fn, in, out, attrs, format);
}

std::string generateOpInfo(const ir::Function &fn,
                           const std::vector<TensorDesc> &inputs,
                           const std::vector<TensorDesc> &outputs, int format) {
    (void)format;
    (void)fn;
    return generateOpInfo(fn, inputs, outputs, {}, format);
}

std::string generateOpInfo(const ir::Function &fn,
                           const std::vector<TensorDesc> &inputs,
                           const std::vector<TensorDesc> &outputs,
                           const std::vector<TensorDesc> &attrs, int format) {
    (void)format;
    const std::string opName = canonicalOpName(fn.name);

    std::ostringstream os;
    os << "{\n";
    os << "  " << quote("op") << ": " << quote(opName) << ",\n";
    os << "  " << quote("input_desc") << ": " << join(inputs, "  ") << ",\n";
    os << "  " << quote("output_desc") << ": " << join(outputs, "  ") << ",\n";
    os << "  " << quote("attr") << ": " << join(attrs, "  ") << ",\n";

    
    
    
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        const std::string name = "input" + std::to_string(i);
        os << "  " << quote(name) << ": {\"name\": \"" << name
           << "\", \"dtype\": \"" << inputs[i].dtype
           << "\", \"format\": \"ND\", \"shape\": [-1], "
              "\"paramType\": \"required\"}";
        os << ",\n";
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        const std::string name = "output" + std::to_string(i);
        os << "  " << quote(name) << ": {\"name\": \"" << name
           << "\", \"dtype\": \"" << outputs[i].dtype
           << "\", \"format\": \"ND\", \"shape\": [-1], "
              "\"paramType\": \"required\"}";
        os << ",\n";
    }
    os << "  " << quote("compute") << ": {\"cost\": \"10\"}";
    os << "}\n";
    return os.str();
}

} 


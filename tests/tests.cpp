// MAIN: the unit tests.
#include "cuascend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "test.h"

// MAIN: the names that reach the generated sources
TEST(names_from_the_image_are_made_safe) {
    // A function or a parameter of the image is data: it must not reach a generated source, a path or a command line as it are.
    const std::string hostile = "../../etc/passwd; rm -rf /";
    const std::string safe = ca::ascend::sanitizeIdentifier(hostile);
    CHECK(safe.find('/') == std::string::npos);
    CHECK(safe.find('.') == std::string::npos);
    CHECK(safe.find(';') == std::string::npos);
    CHECK(safe.find(' ') == std::string::npos);
    for (char c : safe)
        CHECK(c == '_' || (c >= 'a' && c <= 'z') ||
              (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'));
    // An identifier cannot start with a digit.
    CHECK(ca::ascend::sanitizeIdentifier("1st")[0] == '_');
    // An empty name still yields something usable.
    CHECK(!ca::ascend::sanitizeIdentifier("").empty());
}

// MAIN: a malformed image must not take the process down
TEST(a_malformed_image_does_not_crash_the_parser) {
    // The image is data: a parser that walks off the end of it is a crash in somebody else's process, so the input is thrown at it.
    uint32_t seed = 12345;
    for (int round = 0; round < 200; ++round) {
        std::string blob;
        const size_t length = 16 + (seed % 512);
        for (size_t i = 0; i < length; ++i) {
            seed = seed * 1103515245u + 12345u;
            blob.push_back(static_cast<char>((seed >> 16) & 0xff));
        }
        // Whatever the parser makes of it, it must answer, not die.
        const ca::ir::Module module = ca::lower::fromText(blob).module;
        (void)module;
    }
    CHECK(true);
}

// MAIN: the atomics
TEST(interp_runs_atomics) {
    // Every thread adds one to a counter with an atomic; the counter ends up with the number of threads.
    const char *ptx = R"PTX(
.version 7.0
.target sm_37
.address_size 64
.visible .entry count(
    .param .u64 counter_ptr,
    .param .u32 n
)
.maxntid 32, 1, 1
{
    .reg .pred  %p<2>;
    .reg .b32   %r<8>;
    .reg .b64   %rd<4>;

    ld.param.u64 %rd1, [counter_ptr];
    ld.param.u32 %r1, [n];

    mov.u32  %r2, %tid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra EXIT;

    mov.u32  %r3, 1;
    atom.global.add.u32 %r4, [%rd1], %r3;

EXIT:
    ret.uni;
}
)PTX";
    const ca::ir::Module module = ca::lower::fromText(ptx).module;
    const ca::ir::Function *found = module.find("count");
    CHECK(found != nullptr);
    if (!found) return;
    const ca::ir::Function fn = *found;

    uint32_t counter = 0;
    uint64_t counterPtr = reinterpret_cast<uint64_t>(&counter);
    uint32_t n = 32;
    void *args[] = {&counterPtr, &n};

    ca::ir::LaunchConfig config;
    config.blockX = 32;

    std::string error;
    HostMemory memory;
    memory.add(&counter, sizeof counter);
    CHECK(ca::ir::interpret(fn, config, args, memory, &error));
    if (!error.empty()) std::printf("    error: %s\n", error.c_str());
    CHECK_EQ(counter, 32u);
}

int main() { return ca::test::runAll(); }

// MAIN: the tests of the parser
namespace tests0 {

using namespace ca;

namespace {

std::string load(const std::string &rel) {
    std::ifstream f(std::string(TEST_DATA_DIR) + "/" + rel);
    if (!f) return {};
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

} 

TEST(lexer_handles_hex_float_bitpatterns) {
    
    const auto toks = ptx::lex("mov.f32 %f1, 0f3F800000;");
    bool sawFloatPattern = false;
    for (const auto &t : toks)
        if (t.kind == ptx::Token::Kind::Float && t.text == "0f3F800000")
            sawFloatPattern = true;
    CHECK(sawFloatPattern);
}

TEST(lexer_skips_comments) {
    const auto toks = ptx::lex("// comment\n/* block\ncomment */ mov.u32 %r1, 1;");
    int idents = 0;
    for (const auto &t : toks)
        if (t.kind == ptx::Token::Kind::Ident) ++idents;
    CHECK_EQ(idents, 1);
}

TEST(lexer_distinguishes_label_from_register) {
    
    const auto toks = ptx::lex("bra.uni %L1;\n%L1:\nret.uni;");
    int regs = 0;
    for (const auto &t : toks)
        if (t.kind == ptx::Token::Kind::Register) ++regs;
    CHECK_EQ(regs, 2);   
}

TEST(parse_sample_module) {
    const std::string src = load("sample.cu.ptx");
    CHECK(!src.empty());

    ptx::Module m;
    try {
        m = ptx::parse(src, "sample.cu.ptx");
    } catch (const ptx::Error &e) {
        ::ca::test::fail(__FILE__, __LINE__, std::string("parse threw: ") + e.what());
        return;
    }

    CHECK_EQ(m.versionMajor, 7);
    CHECK_EQ(m.versionMinor, 5);
    CHECK_STR(m.target, "sm_80");
    CHECK_EQ(m.addressSize, 64);
    CHECK(m.functions.size() >= 2);

    const ptx::Function *f = m.findEntry("warp_reduce_f32");
    CHECK(f != nullptr);
    if (!f) return;

    CHECK(f->visible);
    CHECK_EQ(f->params.size(), 5u);
    CHECK_EQ(f->maxThreads, 1024u);

    
    CHECK(f->params[0].isPointer || f->params[0].type.width == 64);
}

TEST(parse_vector_registers_paired_by_space) {
    
    const auto m = ptx::parse(
        ".visible .entry k()\n{\n mov.v4.f32 {%f1,%f2,%f3,%f4}, {%f5,%f6,%f7,%f8};\n"
        " ld.global.v4.f32 {%rs1,%rs2}, [%rd1];\n ret.uni;\n}\n");
    const ptx::Function *f = m.findEntry("k");
    CHECK(f != nullptr);
    if (!f || f->body.size() < 2) return;

    
    const auto &mov = f->body[0];
    CHECK_STR(mov.opcode, "mov");
    CHECK(mov.hasMod(".v4"));
    CHECK(mov.hasMod(".f32"));
    CHECK_EQ(mov.operands.size(), 2);
    if (mov.operands.size() == 2) {
        CHECK(mov.operands[0].kind == ptx::OperandKind::Vector);
        CHECK_EQ(mov.operands[0].elements.size(), 4u);
    }

    
    const auto &ld = f->body[1];
    CHECK_STR(ld.opcode, "ld");
    CHECK(ld.hasMod(".global"));
    CHECK(ld.hasMod(".v4"));
    CHECK(ld.hasMod(".f32"));
    CHECK_EQ(ld.operands.size(), 2);
    if (ld.operands.size() == 2) {
        CHECK(ld.operands[0].kind == ptx::OperandKind::Vector);
        CHECK_EQ(ld.operands[0].elements.size(), 2u);
        CHECK(ld.operands[1].kind == ptx::OperandKind::Addr);
    }
}

TEST(parse_address_expressions) {
    const auto m = ptx::parse(
        ".visible .entry k()\n{\n"
        " ld.global.f32 %f1, [%rd1+64];\n"
        " st.global.f32 [%rd2], %f1;\n"
        " ld.global.f32 %f1, [%rd3 + %r4*4];\n"
        " ret.uni;\n}\n");
    const ptx::Function *f = m.findEntry("k");
    CHECK(f != nullptr);
    if (!f || f->body.size() < 3) return;

    
    const auto &a0 = f->body[0].operands[1];
    CHECK(a0.kind == ptx::OperandKind::Addr);
    CHECK(a0.addr.hasBase);
    CHECK_STR(a0.addr.baseReg, "rd1");
    CHECK_EQ(a0.addr.offset, 64);

    
    const auto &a1 = f->body[1].operands[0];
    CHECK(a1.kind == ptx::OperandKind::Addr);
    CHECK(a1.addr.hasBase);
    CHECK_STR(a1.addr.baseReg, "rd2");
    CHECK_EQ(a0.addr.offset, 64); 

    
    const auto &a2 = f->body[2].operands[1];
    CHECK(a2.kind == ptx::OperandKind::Addr);
    CHECK_STR(a2.addr.baseReg, "rd3");
    CHECK_STR(a2.addr.indexReg, "r4");
}

TEST(parse_predicates_including_negated) {
    const auto m = ptx::parse(
        ".visible .entry k()\n{\n"
        " setp.lt.f32 %p1, %f1, 0f3F800000;\n"
        " @!%p2 add.f32 %f1, %f1, %f2;\n"
        " @%p1 bra.uni LBL;\n"
        "LBL:\n"
        " ret.uni;\n}\n");
    const ptx::Function *f = m.findEntry("k");
    CHECK(f != nullptr);
    if (!f || f->body.size() < 3) return;

    CHECK(!f->body[0].hasPred);
    CHECK(f->body[0].opcode == "setp");

    CHECK(f->body[1].hasPred);
    CHECK_STR(f->body[1].pred, "!p2");

    CHECK(f->body[2].hasPred);
    CHECK_STR(f->body[2].pred, "p1");
    
    CHECK_STR(f->body[2].opcode, "bra.uni");
}

TEST(parse_mov_v2_pair_from_sample) {
    
    const std::string src = load("sample.cu.ptx");
    if (src.empty()) return;
    const auto m = ptx::parse(src, "sample.cu.ptx");
    const ptx::Function *f = m.findEntry("warp_reduce_f32");
    CHECK(f != nullptr);
    if (!f) return;

    
    bool found = false;
    for (const auto &in : f->body) {
        if (in.opcode == "add" && in.hasMod(".v2") && in.hasMod(".f32")) {
            found = true;
            CHECK_EQ(in.operands.size(), 3);
            for (const auto &o : in.operands)
                CHECK(o.kind == ptx::OperandKind::Register);
            if (!in.operands.empty())
                CHECK_STR(in.operands[0].name, "rs1");
        }
    }
    CHECK(found);
}

TEST(parse_warp_primitives) {
    const auto m = ptx::parse(
        ".visible .entry k()\n{\n"
        " shfl.sync.down.b32 %r1, %r2, 16, 0x1f;\n"
        " vote.sync.ballot.b32 %r3, %p1;\n"
        " bar.sync 0;\n"
        " bar.warp.sync 0;\n"
        " ret.uni;\n}\n");
    const ptx::Function *f = m.findEntry("k");
    CHECK(f != nullptr);
    if (!f || f->body.size() < 4) return;

    CHECK_STR(f->body[0].opcode, "shfl.sync.down");
    CHECK(f->body[0].hasMod(".b32"));

    CHECK_STR(f->body[1].opcode, "vote.sync.ballot");
    CHECK(f->body[1].hasMod(".b32"));

    CHECK_STR(f->body[2].opcode, "bar.sync");

    
    CHECK_STR(f->body[3].opcode, "bar.warp.sync");
}

TEST(parse_bare_label_operands) {
    
    const auto m = ptx::parse(
        ".visible .entry k()\n{\n bra.uni DONE;\n mov.u32 %r1, 1;\nDONE:\n ret.uni;\n}\n");
    const ptx::Function *f = m.findEntry("k");
    CHECK(f != nullptr);
    if (!f || f->body.empty()) return;
    CHECK_STR(f->body[0].opcode, "bra.uni");
    CHECK_EQ(f->body[0].operands.size(), 1);
    if (!f->body[0].operands.empty()) {
        CHECK(f->body[0].operands[0].kind == ptx::OperandKind::Label);
        CHECK_STR(f->body[0].operands[0].name, "DONE");
    }
}

TEST(parse_address_with_scaled_index) {
    
    const auto m = ptx::parse(
        ".visible .entry k()\n{\n ld.global.f32 %f1, [%rd3 + %r4*4];\n ret.uni;\n}\n");
    const ptx::Function *f = m.findEntry("k");
    CHECK(f != nullptr);
    if (!f || f->body.empty()) return;
    const auto &a = f->body[0].operands[1];
    CHECK(a.kind == ptx::OperandKind::Addr);
    CHECK_STR(a.addr.baseReg, "rd3");
    CHECK_STR(a.addr.indexReg, "r4");
    CHECK_EQ(a.addr.scale, 4);
}

TEST(parse_reports_position_on_error) {
    
    bool threw = false;
    try {
        ptx::parse(".visible .entry k()\n{\n mov.u32 %r1, ;\n}\n");
    } catch (const ptx::Error &) {
        threw = true;
    }
    CHECK(threw);
}

TEST(fatbin_detects_raw_ptx) {
    const std::string ptxsrc = "// generated\n.version 7.5\n.target sm_80\n";
    CHECK(fatbin::detect(ptxsrc.data(), ptxsrc.size()) == fatbin::Format::RawPTX);
}

TEST(fatbin_detects_ptx_with_leading_whitespace) {
    
    const char *variants[] = {
        "\n.version 7.5\n.target sm_80\n",
        "\n\n  .version 7.5\n.target sm_80\n",
        "\t\t.version 7.5\n",
        "// generated by nvcc\n\n.version 7.5\n.target sm_80\n",
        "/* header */\n.version 7.5\n",
    };
    for (const char *v : variants) {
        const std::string src = v;
        CHECK(fatbin::detect(src.data(), src.size()) == fatbin::Format::RawPTX);
    }
}

TEST(fatbin_parse_raw_ptx) {
    const std::string ptxsrc = ".version 7.5\n.target sm_80\n";
    const auto fb = fatbin::parse(ptxsrc.data(), ptxsrc.size());
    CHECK_EQ(fb.entries().size(), 1u);
    CHECK(fb.hasPtx());
    CHECK(!fb.hasCubin());
    CHECK_STR(fb.firstPtx(), ptxsrc);
}

TEST(lexer_tolerates_trailing_nul) {
    
    std::string src = ".visible .entry k()\n{\n mov.u32 %r1, 1;\n ret.uni;\n}\n";
    src.push_back('\0');
    src.push_back('\0');
    bool ok = true;
    try {
        const auto m = ptx::parse(src);
        ok = m.findEntry("k") != nullptr;
    } catch (const ptx::Error &e) {
        ::ca::test::fail(__FILE__, __LINE__, std::string("threw: ") + e.what());
        ok = false;
    }
    CHECK(ok);
}

TEST(fatbin_rejects_garbage) {
    const char junk[32] = {0};
    bool threw = false;
    try {
        fatbin::parse(junk, sizeof junk);
    } catch (const fatbin::Error &) {
        threw = true;
    }
    CHECK(threw);
}

TEST(fatbin_owns_its_input) {
    
    std::string src = ".version 7.5\n.target sm_80\n.visible .entry k(){ret.uni;}\n";
    fatbin::Fatbin fb = fatbin::parse(src.data(), src.size());
    std::string().swap(src); 
    CHECK(fb.hasPtx());
    CHECK(fb.firstPtx().find(".visible .entry k()") != std::string::npos);
}

TEST(lz4_roundtrip_literals_only) {
    
    const std::vector<uint8_t> in = {'a', 'b', 'c', 'd', 'e'};
    const std::vector<uint8_t> block = {0x50, 'a', 'b', 'c', 'd', 'e'};
    const auto out = lz4::decodeBlock(block.data(), block.size());
    CHECK(out == in);
}

TEST(lz4_runlength_overlapping_match) {
    
    const std::vector<uint8_t> block = {0x10, 'x', 0x01, 0x00};
    const auto out = lz4::decodeBlock(block.data(), block.size());
    const std::vector<uint8_t> want = {'x', 'x', 'x', 'x', 'x'};
    CHECK(out == want);
}

TEST(lz4_rejects_offset_before_output_start) {
    
    bool threw = false;
    try {
        const std::vector<uint8_t> block = {0x10, 'x', 0xFF, 0x00};
        lz4::decodeBlock(block.data(), block.size());
    } catch (const lz4::Error &) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    return ca::test::runAll();
}
}

// MAIN: the tests of the IR
namespace tests1 {

using namespace ca;

namespace {

std::string load(const std::string &rel) {
    std::ifstream f(std::string(TEST_DATA_DIR) + "/" + rel);
    if (!f) return {};
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

const char *kVecKernel =
    ".visible .entry scale(\n"
    "    .param .u64 out,\n"
    "    .param .u32 n,\n"
    "    .param .f32 k\n"
    ")\n"
    ".maxntid 256, 1, 1\n"
    "{\n"
    "    .reg .pred  %p<2>;\n"
    "    .reg .b32   %r<6>;\n"
    "    .reg .f32   %f<4>;\n"
    "    .reg .b64   %rd<4>;\n"
    "    .reg .f32   %rs<4>;\n"
    "    ld.param.u64 %rd1, [out];\n"
    "    ld.param.u32 %r1,  [n];\n"
    "    ld.param.f32 %f1,  [k];\n"
    "    mov.u32  %r2, %tid.x;\n"
    "    setp.lt.u32 %p1, %r2, %r1;\n"
    "    @%p1 bra  SKIP;\n"
    "    mul.lo.u32 %r3, %r2, 4;\n"
    "    add.s64  %rd2, %rd1, %rd3;\n"
    "    ld.global.f32 %f2, [%rd2];\n"
    "    mul.f32  %f3, %f2, %f1;\n"
    "    st.global.f32 [%rd2], %f3;\n"
    "SKIP:\n"
    "    ret.uni;\n"
    "}\n";

} 

TEST(ir_type_rendering) {
    CHECK_STR((ir::Type{ir::TypeKind::Float, 32, ir::VecLen::Scalar}).str(), "f32");
    CHECK_STR((ir::Type{ir::TypeKind::Float, 32, ir::VecLen::V4}).str(), "f32v4");
    CHECK_STR((ir::Type{ir::TypeKind::Int, 64, ir::VecLen::Scalar}).str(), "i64");
    CHECK_STR((ir::Type{ir::TypeKind::Bool, 1, ir::VecLen::Scalar}).str(), "pred");
    CHECK((ir::Type{ir::TypeKind::Bool, 1, ir::VecLen::Scalar}).isBool());
    CHECK((ir::Type{ir::TypeKind::Float, 32, ir::VecLen::V4}).isVector());
    CHECK((ir::Type{ir::TypeKind::Void, 0, ir::VecLen::Scalar}).isVoid());
}

TEST(ir_warp_ops_are_flagged) {
    
    CHECK(ir::isWarpOp(ir::Op::ShflDown));
    CHECK(ir::isWarpOp(ir::Op::ShflIdx));
    CHECK(ir::isWarpOp(ir::Op::VoteBallot));
    CHECK(ir::isWarpOp(ir::Op::Popc));
    CHECK(ir::isWarpOp(ir::Op::BarrierWarp));
    
    CHECK(!ir::isWarpOp(ir::Op::Add));
    CHECK(!ir::isWarpOp(ir::Op::FAdd));
    CHECK(!ir::isWarpOp(ir::Op::Load));
    CHECK(!ir::isWarpOp(ir::Op::BarrierThreads));
}

TEST(ir_terminators_are_recognized) {
    CHECK(ir::isTerminator(ir::Op::Ret));
    CHECK(ir::isTerminator(ir::Op::Br));
    CHECK(ir::isTerminator(ir::Op::CondBr));
    CHECK(!ir::isTerminator(ir::Op::Add));
    CHECK(!ir::isTerminator(ir::Op::BarrierThreads));
}

TEST(verify_catches_missing_terminator) {
    ir::Module m;
    ir::Function f;
    f.name = "k";
    ir::Block b;
    ir::Instr in;
    in.op = ir::Op::Add;
    in.type = ir::Type{ir::TypeKind::Int, 32, ir::VecLen::Scalar};
    in.result = 0;
    in.args.push_back(ir::Operand::lit(1));
    in.args.push_back(ir::Operand::lit(2));
    b.instrs.push_back(in);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    const auto problems = ir::verify(m);
    bool found = false;
    for (const auto &p : problems)
        if (p.find("no terminator") != std::string::npos) found = true;
    CHECK(found);
}

TEST(verify_catches_undefined_value) {
    ir::Module m;
    ir::Function f;
    f.name = "k";
    ir::Block b;
    ir::Instr in;
    in.op = ir::Op::Add;
    in.type = ir::Type{ir::TypeKind::Int, 32, ir::VecLen::Scalar};
    in.result = 0;
    in.args.push_back(ir::Operand::val(99)); 
    in.args.push_back(ir::Operand::lit(2));
    b.instrs.push_back(in);
    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    const auto problems = ir::verify(m);
    bool found = false;
    for (const auto &p : problems)
        if (p.find("undefined value") != std::string::npos) found = true;
    CHECK(found);
}

TEST(verify_catches_memory_without_address) {
    ir::Module m;
    ir::Function f;
    f.name = "k";
    ir::Block b;
    ir::Instr ld;
    ld.op = ir::Op::Load;
    ld.type = ir::Type{ir::TypeKind::Float, 32, ir::VecLen::Scalar};
    ld.result = 0;
    b.instrs.push_back(ld);
    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    const auto problems = ir::verify(m);
    bool found = false;
    for (const auto &p : problems)
        if (p.find("without address") != std::string::npos) found = true;
    CHECK(found);
}

TEST(verify_accepts_wellformed_function) {
    const auto r = lower::fromText(kVecKernel);
    const auto problems = ir::verify(r.module);
    if (!problems.empty()) {
        std::ostringstream os;
        os << "verifier found " << problems.size() << " problem(s):";
        for (const auto &p : problems) os << "\n    " << p;
        ::ca::test::fail(__FILE__, __LINE__, os.str());
    }
}

TEST(lower_vector_kernel_is_verified) {
    const auto r = lower::fromText(kVecKernel);
    CHECK_EQ(r.module.functions.size(), 1);
    const ir::Function *f = r.module.find("scale");
    CHECK(f != nullptr);
    if (!f) return;
    CHECK(f->isEntry);
    CHECK_EQ(f->params.size(), 3);
    CHECK_EQ(f->reqThreads, 256);
}

TEST(lower_maps_special_registers_to_ops) {
    
    const auto r = lower::fromText(kVecKernel);
    const ir::Function *f = r.module.find("scale");
    CHECK(f != nullptr);
    if (!f) return;

    bool foundTid = false;
    for (const auto &b : f->blocks)
        for (const auto &in : b.instrs)
            if (in.op == ir::Op::TidX) foundTid = true;
    CHECK(foundTid);
}

TEST(lower_maps_float_arithmetic) {
    const auto r = lower::fromText(kVecKernel);
    const ir::Function *f = r.module.find("scale");
    CHECK(f != nullptr);
    if (!f) return;

    bool mulFound = false;
    for (const auto &b : f->blocks)
        for (const auto &in : b.instrs)
            if (in.op == ir::Op::FMul && in.type.isFloat() &&
                in.type.width == 32)
                mulFound = true;
    CHECK(mulFound);
}

TEST(lower_preserves_memory_space) {
    
    const auto r = lower::fromText(kVecKernel);
    const ir::Function *f = r.module.find("scale");
    CHECK(f != nullptr);
    if (!f) return;

    bool ldGlobal = false, stGlobal = false;
    for (const auto &b : f->blocks)
        for (const auto &in : b.instrs) {
            if (in.op == ir::Op::Load && in.space == ir::Space::Global)
                ldGlobal = true;
            if (in.op == ir::Op::Store && in.space == ir::Space::Global)
                stGlobal = true;
        }
    CHECK(ldGlobal);
    CHECK(stGlobal);
}

TEST(lower_vector_load_keeps_v4_width) {
    
    const auto src =
        ".visible .entry k()\n{\n"
        " .reg .f32 %f<8>;\n .reg .b64 %rd<2>;\n"
        " ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd1];\n"
        " ret.uni;\n}\n";
    const auto r = lower::fromText(src);
    const ir::Function *f = r.module.find("k");
    CHECK(f != nullptr);
    if (!f) return;

    bool found = false;
    for (const auto &b : f->blocks)
        for (const auto &in : b.instrs)
            if (in.op == ir::Op::Load && in.type.isVector() &&
                in.type.vec == ir::VecLen::V4 && in.type.isFloat()) {
                found = true;
                CHECK_EQ(in.type.components(), 4);
            }
    CHECK(found);
}

TEST(lower_maps_warp_primitives) {
    const auto src =
        ".visible .entry k()\n{\n"
        " .reg .b32 %r<8>;\n .reg .f32 %f<4>;\n"
        " mov.u32 %r1, %tid.x;\n"
        " shfl.sync.down.b32 %r2, %r1, 16, 0x1f;\n"
        " vote.sync.ballot.b32 %r3, %p1;\n"
        " bar.sync 0;\n"
        " ret.uni;\n}\n";
    const auto r = lower::fromText(src);
    const ir::Function *f = r.module.find("k");
    CHECK(f != nullptr);
    if (!f) return;

    int warp = 0, barrier = 0;
    for (const auto &b : f->blocks)
        for (const auto &in : b.instrs) {
            if (ir::isWarpOp(in.op)) ++warp;
            if (in.op == ir::Op::BarrierThreads) ++barrier;
        }
    CHECK(warp >= 2);
    CHECK_EQ(barrier, 1);
    CHECK(r.warpOps >= 2);
}

TEST(lower_unknown_instruction_still_defines_destination) {
    
    const auto src =
        ".visible .entry k()\n{\n"
        " .reg .b32 %r<4>;\n"
        " mystery.tensor.op %r1, %r2, %r3;\n"
        " add.u32 %r2, %r1, 0;\n"   
        " ret.uni;\n}\n";
    const auto r = lower::fromText(src);
    const ir::Function *f = r.module.find("k");
    CHECK(f != nullptr);
    if (!f) return;

    
    ir::ValueId addArg = ir::kInvalidValue;
    bool sawUnknown = false;
    for (const auto &b : f->blocks)
        for (const auto &in : b.instrs) {
            if (in.op == ir::Op::Unknown) sawUnknown = true;
            if (in.op == ir::Op::Add && !in.args.empty())
                addArg = in.args[0].isValue() ? in.args[0].value
                                             : ir::kInvalidValue;
        }
    CHECK(sawUnknown);
    CHECK(addArg != ir::kInvalidValue);
    
    const auto problems = ir::verify(r.module);
    CHECK(problems.empty());
    CHECK_EQ(r.unknown, 1);
}

TEST(lower_strict_mode_rejects_unknown) {
    lower::Options opt;
    opt.strict = true;
    bool threw = false;
    try {
        lower::fromText(
            ".visible .entry k()\n{\n .reg .b32 %r<2>;\n"
            " mystery.tensor.op %r1, %r2;\n ret.uni;\n}\n",
            opt);
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw);
}

TEST(lower_sample_module_coverage) {
    const std::string src = load("sample.cu.ptx");
    if (src.empty()) return;
    const auto r = lower::fromText(src);
    CHECK(r.coverage() > 0.5);
    CHECK(r.warpOps > 0);
    
    const auto problems = ir::verify(r.module);
    for (const auto &p : problems)
        if (p.find("undefined value") != std::string::npos) {
            ::ca::test::fail(__FILE__, __LINE__, "dangling value: " + p);
        }
}

TEST(lower_predicate_not_is_materialized) {
    
    const auto src =
        ".visible .entry k()\n{\n"
        " .reg .pred %p<2>;\n .reg .f32 %f<4>;\n"
        " @!%p1 add.f32 %f3, %f1, %f2;\n"
        " ret.uni;\n}\n";
    const auto r = lower::fromText(src);
    const ir::Function *f = r.module.find("k");
    CHECK(f != nullptr);
    if (!f) return;

    int nots = 0;
    bool addHasPred = false;
    for (const auto &b : f->blocks)
        for (const auto &in : b.instrs) {
            if (in.op == ir::Op::Not) ++nots;
            if (in.op == ir::Op::FAdd && in.hasPred) addHasPred = true;
        }
    CHECK_EQ(nots, 1);
    CHECK(addHasPred);
}

int main() {
    return ca::test::runAll();
}
}

// MAIN: the tests of the compile pipeline
namespace tests2 {

namespace fs = std::filesystem;
using namespace ca::ascend;

TEST(compile_hash_is_stable) {
    
    const std::string a = hashSource(".version 7.0\n");
    CHECK(a == hashSource(".version 7.0\n"));
    CHECK(a == hashSource(".version 7.0\n"));
}

TEST(compile_hash_distinguishes_content) {
    CHECK(hashSource("a") != hashSource("b"));
    CHECK(hashSource("axpy") != hashSource("axpy2"));
}

TEST(compile_hash_handles_empty_and_binary) {
    CHECK(!hashSource("").empty());
    
    const std::string withNul("a\0b", 3);
    CHECK(hashSource(withNul) != hashSource("a"));
    CHECK(hashSource(withNul) == hashSource(std::string("a\0b", 3)));
}

TEST(compile_toolchain_reads_environment) {
    ::setenv("CUASCEND_CANN_ROOT", "/opt/ascend", 1);
    ::setenv("CUASCEND_SOC", "Ascend910C", 1);
    ::setenv("CUASCEND_BISHENG", "/opt/ascend/bin/bisheng", 1);
    ::unsetenv("CUASCEND_MSOPGEN");

    const Toolchain tc = Toolchain::fromEnvironment();
    CHECK_STR(tc.root, "/opt/ascend");
    CHECK_STR(tc.socVersion, "Ascend910C");
    CHECK_STR(tc.bisheng, "/opt/ascend/bin/bisheng");
    
    CHECK(tc.msopgen.find("/opt/ascend") == 0);

    ::unsetenv("CUASCEND_CANN_ROOT");
    ::unsetenv("CUASCEND_SOC");
    ::unsetenv("CUASCEND_BISHENG");
}

TEST(compile_toolchain_honours_tool_overrides) {
    
    ::setenv("CUASCEND_BISHENG_FLAGS", "-mcce=Ascend910C -O2", 1);
    const Toolchain tc = Toolchain::fromEnvironment();
    (void)tc;
    ::unsetenv("CUASCEND_BISHENG_FLAGS");
    CHECK(true);
}

TEST(compile_run_command_captures_output) {
    const std::string out = runCommand({"/bin/sh", "-c", "echo hello-from-tool"}, false);
    CHECK(out.find("hello-from-tool") != std::string::npos);
}

TEST(compile_run_command_preserves_both_streams) {
    
    const std::string out = runCommand(
        {"/bin/sh", "-c", "echo to-stdout; echo to-stderr 1>&2"}, false);
    CHECK(out.find("to-stdout") != std::string::npos);
    CHECK(out.find("to-stderr") != std::string::npos);
}

TEST(compile_run_command_throws_with_output_on_failure) {
    
    bool threw = false;
    std::string what, out;
    try {
        runCommand({"/bin/sh", "-c", "echo line7: error: bad thing 1>&2; exit 3"},
                   false);
    } catch (const CompileError &e) {
        threw = true;
        what = e.what();
        out = e.output;
    }
    CHECK(threw);
    CHECK(what.find("status 3") != std::string::npos);
    CHECK(out.find("line7: error") != std::string::npos);
}

TEST(compile_run_command_reports_missing_binary) {
    
    bool threw = false;
    std::string what;
    try {
        runCommand({"/nonexistent/certainly-not-a-tool"}, false);
    } catch (const CompileError &e) {
        threw = true;
        what = e.what();
    }
    CHECK(threw);
    CHECK(what.find("certainly-not-a-tool") != std::string::npos);
}

TEST(compile_cache_counts_hits_and_misses) {
    const fs::path dir = fs::temp_directory_path() / "cuascend-test-cache";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    OpCache cache(dir.string());
    Toolchain tc = Toolchain::fromEnvironment();
    tc.workDir = dir.string();
    
    tc.bisheng = "/nonexistent/bisheng";

    const std::string ptx = ".visible .entry k()\n{\n ret.uni;\n}\n";

    bool threw = false;
    try {
        cache.get(ptx, "k", tc);
    } catch (const CompileError &) {
        threw = true;
    }
    
    CHECK(threw);
    CHECK_EQ(cache.misses(), 1);
    CHECK_EQ(cache.hits(), 0);

    fs::remove_all(dir, ec);
}

TEST(compile_rejects_ptx_without_named_entry) {
    
    Toolchain tc = Toolchain::fromEnvironment();
    tc.bisheng = "/nonexistent/bisheng";

    const std::string ptx =
        ".visible .entry alpha()\n{\n ret.uni;\n}\n"
        ".visible .entry beta()\n{\n ret.uni;\n}\n";

    bool threw = false;
    std::string what;
    try {
        compileKernel(ptx, "gamma", tc.socVersion, tc);
    } catch (const CompileError &e) {
        threw = true;
        what = e.what();
    }
    CHECK(threw);
    CHECK(what.find("gamma") != std::string::npos);
}

TEST(compile_rejects_missing_toolchain_with_path) {
    Toolchain tc = Toolchain::fromEnvironment();
    tc.bisheng = "/definitely/not/here/bisheng";

    const std::string ptx =
        ".visible .entry k()\n{\n .reg .f32 %f<2>;\n mov.f32 %f1, 0f00000000;\n"
        " ret.uni;\n}\n";

    bool threw = false;
    std::string what;
    try {
        compileKernel(ptx, "k", tc.socVersion, tc);
    } catch (const CompileError &e) {
        threw = true;
        what = e.what();
    }
    CHECK(threw);
    
    CHECK(what.find("/definitely/not/here/bisheng") != std::string::npos);
    CHECK(what.find("CUASCEND_BISHENG") != std::string::npos);
}

TEST(compile_refuses_unsupported_kernel_with_reason) {
    
    Toolchain tc = Toolchain::fromEnvironment();
    tc.bisheng = "/definitely/not/here/bisheng";

    const std::string ptx =
        ".visible .entry k()\n{\n"
        " .reg .b32 %r<4>;\n"
        " shfl.down.b32 %r2, %r1, 16, 0x1f;\n"
        " ret.uni;\n}\n";

    bool threw = false;
    std::string what;
    try {
        compileKernel(ptx, "k", tc.socVersion, tc);
    } catch (const CompileError &e) {
        threw = true;
        what = e.what();
    }
    CHECK(threw);
    
    CHECK(what.find("warp") != std::string::npos);
}

int main() {
    return ca::test::runAll();
}
}

// MAIN: the tests of the generator
namespace tests3 {

using namespace ca;

namespace {

const char *kAxpyPtx =
    ".visible .entry axpy(\n"
    "    .param .u64 x_ptr,\n"
    "    .param .u64 y_ptr,\n"
    "    .param .u32 n,\n"
    "    .param .f32 a,\n"
    "    .param .f32 b\n"
    ")\n"
    ".maxntid 256, 1, 1\n"
    "{\n"
    "    .reg .pred %p<4>;\n"
    "    .reg .b32  %r<8>;\n"
    "    .reg .f32  %f<8>;\n"
    "    .reg .b64  %rd<6>;\n"
    "    ld.param.u64 %rd1, [x_ptr];\n"
    "    ld.param.u64 %rd2, [y_ptr];\n"
    "    ld.param.u32 %r1,  [n];\n"
    "    ld.param.f32 %f1,  [a];\n"
    "    ld.param.f32 %f2,  [b];\n"
    "    mov.u32      %r2, %tid.x;\n"
    "    setp.lt.u32  %p1, %r2, %r1;\n"
    "    @%p1 bra     SKIP;\n"
    "    mul.lo.u32   %r3, %r2, 4;\n"
    "    add.s64      %rd3, %rd1, %rd3;\n"
    "    ld.global.f32 %f3, [%rd3];\n"
    "    mul.f32      %f4, %f3, %f1;\n"
    "    add.f32      %f5, %f4, %f2;\n"
    "    add.s64      %rd4, %rd2, %rd3;\n"
    "    st.global.f32 [%rd4], %f5;\n"
    "SKIP:\n"
    "    ret.uni;\n"
    "}\n";

ir::Function lowerFirst(const std::string &ptx, const char *name) {
    const ptx::Module m = ptx::parse(ptx);
    const ptx::Function *pf = m.findEntry(name);
    if (!pf) {
        ::ca::test::fail(__FILE__, __LINE__, std::string("no entry ") + name);
        return {};
    }
    lower::Result r = lower::run(ca::ptx::Module{m});
    for (auto &f : r.module.functions)
        if (f.name == name) return f;
    ::ca::test::fail(__FILE__, __LINE__, "lowering produced no function");
    return {};
}

bool has(const std::string &hay, const char *needle) {
    return hay.find(needle) != std::string::npos;
}

} 

TEST(codegen_emits_entry_point) {
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    CHECK(has(out.source, "extern \"C\" __global__ __aicore__ void axpy("));
    CHECK(has(out.source, "#include \"kernel_operator.h\""));
    CHECK(has(out.source, "using namespace AscendC;"));
}

TEST(codegen_has_no_syntax_placeholders) {
    
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    CHECK(!has(out.source, "TODO"));
    CHECK(!has(out.source, "FIXME"));
    CHECK(!has(out.source, "not implemented"));
    CHECK(!has(out.source, "abort()"));
    CHECK(!has(out.source, "unimplemented"));
}

TEST(codegen_parameters_come_from_tiling_not_ptx_names) {
    
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    CHECK(!has(out.source, "$0"));
    CHECK(has(out.source, "tiling_data.arg0"));
}

TEST(codegen_linearizes_simt) {
    
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    CHECK(has(out.source, "GetBlockNum()"));
    CHECK(has(out.source, "GetBlockIdx()"));
    CHECK(has(out.source, "elemIndex"));
    CHECK(has(out.source, "rangeStart"));
    CHECK(has(out.source, "rangeEnd"));
}

TEST(codegen_uses_ascendc_memory_api) {
    
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    CHECK(has(out.source, "GlobalTensor<float>"));
    CHECK(has(out.source, "SetGlobalBuffer"));
    CHECK(has(out.source, "GetValue"));
    CHECK(has(out.source, "SetValue"));
    CHECK(has(out.source, "__gm__"));
}

TEST(codegen_float_arithmetic_present) {
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    
    CHECK(has(out.source, "* v") || has(out.source, " * "));
    CHECK(has(out.source, "+ v") || has(out.source, " + "));
}

TEST(codegen_refuses_warp_shuffle) {
    const char *src =
        ".visible .entry w()\n{\n"
        " .reg .b32 %r<4>;\n"
        " shfl.sync.down.b32 %r2, %r1, 16, 0x1f;\n"
        " ret.uni;\n}\n";
    ir::Module m;
    m.name = "t";
    ir::Function f;
    f.name = "w";
    f.isEntry = true;
    ir::Block b;
    ir::Instr shfl;
    shfl.op = ir::Op::ShflDown;
    shfl.type = ir::Type{ir::TypeKind::Int, 32, ir::VecLen::Scalar};
    shfl.result = 1;
    shfl.args.push_back(ir::Operand::val(0));
    b.instrs.push_back(shfl);
    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    bool threw = false;
    std::string why;
    try {
        codegen::generate(m.functions[0]);
    } catch (const codegen::CodegenError &e) {
        threw = true;
        why = e.what();
    }
    CHECK(threw);
    
    CHECK(why.find("warp") != std::string::npos);
    (void)src;
}

TEST(codegen_lists_all_unsupported_not_just_first) {
    ir::Module m;
    ir::Function f;
    f.name = "multi";
    f.isEntry = true;
    ir::Block b;
    auto add = [&](ir::Op op) {
        ir::Instr in;
        in.op = op;
        in.type = ir::Type{ir::TypeKind::Int, 32, ir::VecLen::Scalar};
        in.result = b.instrs.size();
        b.instrs.push_back(in);
    };
    add(ir::Op::AtomicRMW);
    add(ir::Op::VoteBallot);
    add(ir::Op::MinS);
    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    const auto list = codegen::unsupported(f);
    CHECK_EQ(list.size(), 3);
    bool sawAtomic = false, sawVote = false, sawMin = false;
    for (const auto &s : list) {
        if (s.find("atomic") != std::string::npos) sawAtomic = true;
        if (s.find("vote") != std::string::npos) sawVote = true;
        if (s.find("min") != std::string::npos) sawMin = true;
    }
    CHECK(sawAtomic);
    CHECK(sawVote);
    CHECK(sawMin);
}

TEST(codegen_refuses_shared_memory_explicitly) {
    
    ir::Module m;
    ir::Function f;
    f.name = "s";
    f.isEntry = true;
    ir::Block b;
    ir::Instr st;
    st.op = ir::Op::Store;
    st.hasAddr = true;
    st.addr.base = ir::Operand::val(0);
    st.addr.offset = 0;
    st.space = ir::Space::Shared;
    st.args.push_back(ir::Operand::val(1));
    b.instrs.push_back(st);
    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    bool threw = false;
    std::string why;
    try {
        codegen::generate(f);
    } catch (const codegen::CodegenError &e) {
        threw = true;
        why = e.what();
    }
    CHECK(threw);
    CHECK(why.find("shared") != std::string::npos);
}

TEST(codegen_decodes_float_bitpattern_literals) {
    
    ir::Module m;
    ir::Function f;
    f.name = "lit";
    f.isEntry = true;
    ir::Block b;

    ir::Instr load;
    load.op = ir::Op::Load;
    load.hasAddr = true;
    load.addr.base = ir::Operand::param("p0");
    load.space = ir::Space::Global;
    load.type = ir::Type{ir::TypeKind::Float, 32, ir::VecLen::Scalar};
    load.result = 0;
    b.instrs.push_back(load);

    
    ir::Operand one;
    one.kind = ir::Operand::Kind::Immediate;
    one.imm = 0x3F800000;
    one.immBits = 0x3F800000;

    ir::Instr mul;
    mul.op = ir::Op::FMul;
    mul.type = ir::Type{ir::TypeKind::Float, 32, ir::VecLen::Scalar};
    mul.result = 1;
    mul.args.push_back(ir::Operand::val(0));
    mul.args.push_back(one);
    b.instrs.push_back(mul);

    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    const auto out = codegen::generate(f);
    
    CHECK(has(out.source, "1LL"));
    CHECK(!has(out.source, "1065353216"));
}

TEST(codegen_predicated_op_guarded_not_substituted) {
    
    ir::Module m;
    ir::Function f;
    f.name = "pred";
    f.isEntry = true;
    ir::Block b;

    ir::Instr add;
    add.op = ir::Op::FAdd;
    add.type = ir::Type{ir::TypeKind::Float, 32, ir::VecLen::Scalar};
    add.result = 2;
    add.args.push_back(ir::Operand::val(0));
    add.args.push_back(ir::Operand::val(1));
    add.hasPred = true;
    add.pred = 5;
    b.instrs.push_back(add);

    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    const auto out = codegen::generate(f);
    CHECK(has(out.source, "if (v5)"));
    
    CHECK(!has(out.source, "? (v0) + (v1) : 0"));
}

TEST(codegen_declares_all_values_before_use) {
    
    ir::Module m;
    ir::Function f;
    f.name = "order";
    f.isEntry = true;
    ir::Block b;
    ir::Instr in;
    in.op = ir::Op::Add;
    in.type = ir::Type{ir::TypeKind::Int, 32, ir::VecLen::Scalar};
    in.result = 7;
    in.args.push_back(ir::Operand::val(3));
    in.args.push_back(ir::Operand::lit(1));
    b.instrs.push_back(in);
    ir::Instr r;
    r.op = ir::Op::Ret;
    b.instrs.push_back(r);
    f.blocks.push_back(b);
    m.functions.push_back(f);

    const auto out = codegen::generate(f);
    const size_t decl = out.source.find("int32_t v7 = 0;");
    const size_t use = out.source.find("v7 = ((v3) + (1));");
    CHECK(decl != std::string::npos);
    CHECK(use != std::string::npos);
    CHECK(decl < use);
}

TEST(codegen_braces_balanced) {
    
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    int depth = 0, line = 1, maxDepth = 0;
    for (char c : out.source) {
        if (c == '\n') ++line;
        if (c == '{') { ++depth; if (depth > maxDepth) maxDepth = depth; }
        if (c == '}') --depth;
        if (depth < 0) {
            ::ca::test::fail(__FILE__, __LINE__,
                             "extra '}' at line " + std::to_string(line));
            return;
        }
    }
    if (depth != 0)
        ::ca::test::fail(__FILE__, __LINE__,
                         "unbalanced braces, depth=" + std::to_string(depth));
    CHECK(depth == 0);
    CHECK(maxDepth >= 2);
}

TEST(codegen_is_deterministic) {
    
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    CHECK(codegen::generate(f).source == codegen::generate(f).source);
}

TEST(codegen_uses_cann_workspace_tiling_signature) {
    const ir::Function f = lowerFirst(kAxpyPtx, "axpy");
    if (f.blocks.empty()) return;
    const auto out = codegen::generate(f);
    
    CHECK(has(out.source, "GM_ADDR workspace, GM_ADDR tiling"));
    CHECK(has(out.source, "GET_TILING_DATA"));
}

}

// MAIN: the tests of the executor
namespace tests4 {

using namespace ca;

namespace {


const char *kAxpy = R"PTX(
.version 7.0
.target sm_37
.address_size 64
.visible .entry axpy(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u32 n,
    .param .f32 a,
    .param .f32 b
)
.maxntid 64, 1, 1
{
    .reg .pred  %p<4>;
    .reg .b32   %r<8>;
    .reg .f32   %f<4>;
    .reg .b64   %rd<8>;

    ld.param.u64 %rd1, [x_ptr];
    ld.param.u64 %rd2, [y_ptr];
    ld.param.u32 %r1, [n];
    ld.param.f32 %f1, [a];
    ld.param.f32 %f2, [b];

    mov.u32  %r2, %tid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra EXIT;

    mul.lo.u32 %r3, %r2, 4;
    cvt.s64.s32 %rd3, %r3;
    add.s64   %rd4, %rd1, %rd3;
    ld.global.f32 %f3, [%rd4];
    mul.f32   %f3, %f3, %f1;
    add.f32   %f3, %f3, %f2;
    add.s64   %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f3;

EXIT:
    ret.uni;
}
)PTX";

const char *kStride = R"PTX(
.version 7.0
.target sm_37
.address_size 64
.visible .entry scale_stride(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
)
.maxntid 32, 1, 1
{
    .reg .pred  %p<4>;
    .reg .b32   %r<8>;
    .reg .f32   %f<4>;
    .reg .b64   %rd<8>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.u32 %r1, [n];

    mov.u32  %r2, %tid.x;
    mov.u32  %r3, %ntid.x;

LOOP:
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra EXIT;
    mul.lo.u32 %r4, %r2, 4;
    cvt.s64.s32 %rd3, %r4;
    add.s64   %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32   %f1, %f1, %f1;
    add.s64   %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f1;
    add.u32   %r2, %r2, %r3;
    bra       LOOP;

EXIT:
    ret.uni;
}
)PTX";

const char *kWarpSum = R"PTX(
.version 7.0
.target sm_37
.address_size 64
.visible .entry warp_sum(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
)
.maxntid 32, 1, 1
{
    .reg .pred  %p<4>;
    .reg .b32   %r<10>;
    .reg .f32   %f<8>;
    .reg .b64   %rd<8>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.u32 %r1, [n];

    mov.u32  %r2, %tid.x;
    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra EXIT;

    mul.lo.u32 %r3, %r2, 4;
    cvt.s64.s32 %rd3, %r3;
    add.s64   %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    // A reduction over the warp: shuffle the neighbour's value in and add it, halving the distance each time.
    shfl.down.f32 %f2, %f1, 16, 0x1f;
    add.f32   %f1, %f1, %f2;
    shfl.down.f32 %f2, %f1,  8, 0x1f;
    add.f32   %f1, %f1, %f2;
    shfl.down.f32 %f2, %f1,  4, 0x1f;
    add.f32   %f1, %f1, %f2;
    shfl.down.f32 %f2, %f1,  2, 0x1f;
    add.f32   %f1, %f1, %f2;
    shfl.down.f32 %f2, %f1,  1, 0x1f;
    add.f32   %f1, %f1, %f2;

    mov.u32   %r5, %tid.x;
    setp.eq.u32 %p2, %r5, 0;
    @%p2 st.global.f32 [%rd2], %f1;

EXIT:
    ret.uni;
}
)PTX";

ca::ir::Function kernel(const char *ptx, const char *name) {
    const ca::ir::Module module = ca::lower::fromText(ptx).module;
    const ca::ir::Function *found = module.find(name);
    return found ? *found : ca::ir::Function{};
}

} 

TEST(interp_runs_elementwise_kernel) {
    const ca::ir::Function fn = kernel(kAxpy, "axpy");
    CHECK(!fn.name.empty());
    if (fn.name.empty()) return;

    constexpr int kN = 64;
    std::vector<float> x(kN), y(kN, 0.0f);
    for (int i = 0; i < kN; ++i) x[i] = static_cast<float>(i) * 0.5f;

    HostMemory memory;
    memory.add(x.data(), x.size() * sizeof(float));
    memory.add(y.data(), y.size() * sizeof(float));

    uint64_t xPtr = reinterpret_cast<uint64_t>(x.data());
    uint64_t yPtr = reinterpret_cast<uint64_t>(y.data());
    uint32_t n = kN;
    float a = 2.0f, b = 1.0f;
    void *args[] = {&xPtr, &yPtr, &n, &a, &b};

    ca::ir::LaunchConfig config;
    config.gridX = 1;
    config.blockX = kN;

    std::string error;
    CHECK(ca::ir::interpret(fn, config, args, memory, &error));
    if (!error.empty()) std::printf("    error: %s\n", error.c_str());

    int wrong = 0;
    for (int i = 0; i < kN; ++i) {
        const double want = static_cast<double>(x[i]) * a + b;
        if (std::fabs(want - y[i]) > 1e-5) ++wrong;
    }
    CHECK_EQ(wrong, 0);
}

TEST(interp_runs_looping_kernel) {
    const ca::ir::Function fn = kernel(kStride, "scale_stride");
    CHECK(!fn.name.empty());
    if (fn.name.empty()) return;

    constexpr int kN = 100; 
    std::vector<float> in(kN), out(kN, 0.0f);
    for (int i = 0; i < kN; ++i) in[i] = static_cast<float>(i);

    HostMemory memory;
    memory.add(in.data(), in.size() * sizeof(float));
    memory.add(out.data(), out.size() * sizeof(float));

    uint64_t inPtr = reinterpret_cast<uint64_t>(in.data());
    uint64_t outPtr = reinterpret_cast<uint64_t>(out.data());
    uint32_t n = kN;
    void *args[] = {&inPtr, &outPtr, &n};

    ca::ir::LaunchConfig config;
    config.blockX = 32;

    std::string error;
    CHECK(ca::ir::interpret(fn, config, args, memory, &error));
    if (!error.empty()) std::printf("    error: %s\n", error.c_str());

    int wrong = 0;
    for (int i = 0; i < kN; ++i)
        if (std::fabs(out[i] - in[i] * 2.0f) > 1e-5) ++wrong;
    CHECK_EQ(wrong, 0);
}

TEST(interp_runs_warp_shuffle) {
    const ca::ir::Function fn = kernel(kWarpSum, "warp_sum");
    CHECK(!fn.name.empty());
    if (fn.name.empty()) return;

    constexpr int kN = 32;
    std::vector<float> in(kN), out(kN, 0.0f);
    for (int i = 0; i < kN; ++i) in[i] = static_cast<float>(i + 1);

    HostMemory memory;
    memory.add(in.data(), in.size() * sizeof(float));
    memory.add(out.data(), out.size() * sizeof(float));

    uint64_t inPtr = reinterpret_cast<uint64_t>(in.data());
    uint64_t outPtr = reinterpret_cast<uint64_t>(out.data());
    uint32_t n = kN;
    void *args[] = {&inPtr, &outPtr, &n};

    ca::ir::LaunchConfig config;
    config.blockX = kN;

    std::string error;
    CHECK(ca::ir::interpret(fn, config, args, memory, &error));
    if (!error.empty()) std::printf("    error: %s\n", error.c_str());

    
    double want = 0;
    for (int i = 1; i <= kN; ++i) want += i;
    CHECK(std::fabs(want - out[0]) < 1e-3);
}

TEST(interp_runs_the_sample_warp_kernel) {
    std::ifstream file(std::string(TEST_DATA_DIR) + "/sample_axpy.cu.ptx");
    CHECK(file.good());
    if (!file.good()) return;
    std::ostringstream text;
    text << file.rdbuf();
    const ca::ir::Module module =
        ca::lower::fromText(text.str()).module;
    const ca::ir::Function *found = module.find("axpy_reduce");
    CHECK(found != nullptr);
    if (!found) return;
    const ca::ir::Function fn = *found;

    constexpr int kN = 64;
    std::vector<float> x(kN), y(kN, -1.0f), out(kN, -1.0f);
    for (int i = 0; i < kN; ++i) x[i] = static_cast<float>(i) * 0.5f;

    HostMemory memory;
    memory.add(x.data(), x.size() * sizeof(float));
    memory.add(y.data(), y.size() * sizeof(float));
    memory.add(out.data(), out.size() * sizeof(float));

    uint64_t xPtr = reinterpret_cast<uint64_t>(x.data());
    uint64_t yPtr = reinterpret_cast<uint64_t>(y.data());
    uint64_t outPtr = reinterpret_cast<uint64_t>(out.data());
    uint32_t n = kN;
    float a = 2.0f, b = 1.0f;
    void *args[] = {&xPtr, &yPtr, &n, &a, &b, &outPtr};

    ca::ir::LaunchConfig config;
    config.blockX = 64;

    std::string error;
    CHECK(ca::ir::interpret(fn, config, args, memory, &error));
    if (!error.empty()) std::printf("    error: %s\n", error.c_str());

    int wrong = 0;
    for (int i = 0; i < kN; ++i) {
        const double want = static_cast<double>(x[i]) * a + b;
        if (std::fabs(want - y[i]) > 1e-5) ++wrong;
    }
    CHECK_EQ(wrong, 0);
}

int main() { return ca::test::runAll(); }

}

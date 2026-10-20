// A minimal test framework with no external dependencies: the project must build anywhere.
#ifndef CA_TEST_H
#define CA_TEST_H

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "cuascend.h"

#include <cstdint>
#include <map>

namespace ca::test {

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case> &registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(const char *name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int &failures() {
    static int f = 0;
    return f;
}

inline void fail(const char *file, int line, const std::string &msg) {
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, msg.c_str());
    ++failures();
}

inline int runAll() {
    int failed = 0;
    for (auto &c : registry()) {
        const int before = failures();
        c.fn();
        const bool ok = failures() == before;
        std::fprintf(stderr, "%s %s\n", ok ? "  ok  " : "  FAIL", c.name.c_str());
        if (!ok) ++failed;
    }
    std::fprintf(stderr, "\n%zu test(s), %d failed\n", registry().size(), failed);
    return failed == 0 ? 0 : 1;
}

} // namespace ca::test

// Host memory the executor can read and write: the buffers are the caller's own allocations, registered by address.
class HostMemory : public ca::ir::Memory {
public:
    void add(void *pointer, size_t bytes) {
        buffers_[reinterpret_cast<uint64_t>(pointer)] = {pointer, bytes};
    }

    bool view(uint64_t address, ca::ir::HostView &out) override {
        for (auto &entry : buffers_) {
            if (address >= entry.first &&
                address < entry.first + entry.second.bytes) {
                out.base = entry.first;
                out.data = static_cast<uint8_t *>(entry.second.pointer);
                out.size = entry.second.bytes;
                return true;
            }
        }
        return false;
    }

    void flush() override {}

private:
    struct Buffer {
        void *pointer;
        size_t bytes;
    };
    std::map<uint64_t, Buffer> buffers_;
};

#define TEST(name)                                                             \
    static void name();                                                        \
    static ::ca::test::Registrar reg_##name(#name, name);                      \
    static void name()

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond))                                                           \
            ::ca::test::fail(__FILE__, __LINE__, "CHECK(" #cond ")");          \
    } while (0)

#define CHECK_EQ(a, b)                                                         \
    do {                                                                       \
        const long long va_ = static_cast<long long>(a);                       \
        const long long vb_ = static_cast<long long>(b);                       \
        if (!(va_ == vb_)) {                                                   \
            std::ostringstream os_;                                            \
            os_ << #a " == " #b " (" << va_ << " vs " << vb_ << ")";          \
            ::ca::test::fail(__FILE__, __LINE__, os_.str());                   \
        }                                                                      \
    } while (0)

#define CHECK_STR(a, b)                                                        \
    do {                                                                       \
        const std::string va_ = (a);                                           \
        const std::string vb_ = (b);                                           \
        if (va_ != vb_)                                                        \
            ::ca::test::fail(__FILE__, __LINE__,                                \
                             std::string(#a " != " #b "\n  got: ") + va_ +     \
                                 "\n  want: " + vb_);                         \
    } while (0)

#endif // CA_TEST_H
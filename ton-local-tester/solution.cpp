/*
 * solution.cpp
 *
 * Example solution.
 * This is (almost) how blocks are actually compressed in TON.
 * Normally, blocks are stored using vm::std_boc_serialize with mode=31.
 * Compression algorithm takes a block, converts it to mode=2 (which has less extra information) and compresses it using lz4.
 */
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <optional>
#include <vector>
#include <bitset>
#include <queue>
#include <cmath>
#include <numeric>
#include "td/utils/lz4.h"
#include "td/utils/base64.h"
#include "vm/boc.h"
#include "block/block-auto.h"
#include "crypto/vm/boc-writers.h"
#include "tdutils/td/utils/misc.h"
#include "tdutils/td/utils/Gzip.h"

std::map<std::string, std::map<int, int>> byte_cnt;

void add_int(const std::string& name, int32_t value) {
  #ifndef ONLINE_JUDGE
    ++byte_cnt[name][value];
  #endif
}

void add_char(const std::string& name, unsigned char value) {
  add_int(name, value);
}

#if __cplusplus >= 201703L
  namespace debug {
    using namespace std;
    namespace TypeTraits {
        template<class T> constexpr bool IsString = false;
        template<> constexpr bool IsString<string> = true;
        template<class T, class = void> struct IsIterableStruct : false_type{};
        template<class T>
        struct IsIterableStruct<
            T,
            void_t<
                decltype(begin(declval<T>())),
                decltype(end(declval<T>()))
            >
        > : true_type{};
        template<class T> constexpr bool IsIterable = IsIterableStruct<T>::value;
        template<class T> constexpr bool NonStringIterable = !IsString<T> && IsIterable<T>;
        template<class T> constexpr bool DoubleIterable = IsIterable<decltype(*begin(declval<T>()))>;
    };
    // Declaration (for cross-recursion)
    template<class T>
    auto pdbg(const T&x) -> enable_if_t<!TypeTraits::NonStringIterable<T>, string>;
    string pdbg(const string &x);
    template<class T>
    auto pdbg(const T &x) -> enable_if_t<TypeTraits::NonStringIterable<T>, string>;
    template<class T, class U>
    string pdbg(const pair<T, U> &x);

    // Implementation
    template<class T>
    auto pdbg(const T &x) -> enable_if_t<!TypeTraits::NonStringIterable<T>, string> {
        stringstream ss;
        ss << x;
        return ss.str();
    }
    template<class T, class U>
    string pdbg(const pair<T, U> &x) {
        return "{" + pdbg(x.first) + "," + pdbg(x.second) + "}";
    }
    string pdbg(const string &x) {
        return "\"" + x + "\"";
    }
    template<class T>
    auto pdbg(const T &x) -> enable_if_t<TypeTraits::NonStringIterable<T>, string> {
        auto begin = x.begin();
        auto end = x.end();
        string del = "";
        if (TypeTraits::DoubleIterable<T>) {
            del = "\n";
        }
        string ans;
        ans += "{" + del;
        if (begin != end) ans += pdbg(*begin++);
        while (begin != end) {
            ans += "," + del + pdbg(*begin++);
        }
        ans += del + "}";
        return ans;
    }
    template<class T> string dbgout(const T &x) { return pdbg(x); }
    template<class T, class... U>
    string dbgout(T const &t, const U &... u) {
        string ans = pdbg(t);
        ans += ", ";
        ans += dbgout(u...);
        return ans;
    }
  };
#endif

namespace log_level {
  enum class LOG_LEVEL {
    ALWAYS = 0, // Logs that always happen
    ONCE = 1,   // Logs, that happen once per program
    CELL = 2,   // Logs, that happen once per cell
    BYTE = 3,   // Logs, that happen once per each byte of data
    BIT = 4,    // Logs, that happen once per each bit of data
    SKIP = 1000 // Logs, that are never written
  };

  static constexpr enum LOG_LEVEL global_log_level = LOG_LEVEL::ALWAYS;

  static constexpr auto ENCODER_STAT = LOG_LEVEL::SKIP;
  static constexpr auto ENCODER_DATA = LOG_LEVEL::SKIP;
  static constexpr auto BIT_IO = LOG_LEVEL::BIT;
  static constexpr auto NUMBER = LOG_LEVEL::BYTE;
  static constexpr auto CELL_META = LOG_LEVEL::CELL;
  static constexpr auto COMPRESSION_META = LOG_LEVEL::ONCE;

  constexpr bool check_log_level(enum LOG_LEVEL log_level) {
    return int(global_log_level) >= int(log_level);
  }
} // namespace log_level

#ifndef ONLINE_JUDGE
    void flush() { std::cerr << std::flush; }
    void flushln() { std::cerr << std::endl; }
    template<class T> void print(const T &x) { std::cerr << x; }
    template<class T, class ...U> void print(const T &x, const U&... u) { print(x); print(u...); }
    template<class ...T> void println(const T&... u) { print(u..., '\n'); }
    #define dbg(...) print("[", #__VA_ARGS__, "] = ", debug::dbgout(__VA_ARGS__)), flushln()
    #define msg(...) print("[", __VA_ARGS__, "]"), flushln()
    #define DBG(level, ...) if (log_level::check_log_level(level)) dbg(__VA_ARGS__)
    #define MSG(level, ...) if (log_level::check_log_level(level)) msg(__VA_ARGS__)
#else
    #define dbg(...) 0
    #define msg(...) 0
    #define DBG(level, ...) 0
    #define MSG(level, ...) 0
#endif

uint8_t* get_buffer_slice_data(const td::BufferSlice& slice) {
  return const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(slice.data()));
}

namespace BWT {

using namespace std;

template<class T>
inline int sz(const T &x) { return x.size(); }
#define all(a) a.begin(), a.end()
#define pb push_back
using pii = pair<int, int>;
template<class T>
inline void sort(T &x) { sort(all(x)); }

template<class T>
vector<int> suffixarray(T s) {
    vector<int> val(all(s));
    auto x = val;
    sort(x);
    x.resize(unique(all(x)) - x.begin());
    for (auto &i : val)
        i = lower_bound(all(x), i) - x.begin();
    val.pb(-1);
    vector<int> p(sz(val));
    for (int i = 0; i < sz(p); ++i) p[i] = i + 1;
    p[sz(p) - 1] = sz(p) - 1;
    int lg = 0;
    vector<int> ans(sz(s));
    for (int i = 0; i < sz(ans); ++i) ans[i] = i;
    sort(all(ans), [&](int i, int j) {
        return val[i] < val[j];
    });
    while ((1 << lg) < sz(val)) {
        ++lg;
        int past = 0;
        for (int i = 0; i < sz(ans); ++i)
            if (val[ans[i]] != val[ans[past]]) {
                sort(ans.begin() + past, ans.begin() + i, [&](int i, int j) {
                    return pii(val[i], val[p[i]]) < pii(val[j], val[p[j]]);
                });
                past = i;
            }
        sort(ans.begin() + past, ans.end(), [&](int i, int j) {
            return pii(val[i], val[p[i]]) < pii(val[j], val[p[j]]);
        });
        vector<pii> coord;
        for (auto i : ans) coord.pb({val[i], val[p[i]]});
        coord.resize(unique(all(coord)) - coord.begin());
        int ptr = 0;
        vector<int> newval(sz(ans));
        newval.pb(-1);
        for (auto i : ans) {
            while (coord[ptr] < pii(val[i], val[p[i]])) ++ptr;
            newval[i] = ptr;
        }
        val = newval;
        for (int i = 0; i < sz(p); ++i)
            p[i] = p[p[i]];
    }
    return ans;
}

using data_type = int;
using byte_buffer = std::vector<data_type>;
static constexpr data_type SPECIAL_SYMBOL = -1;

byte_buffer to_byte_buffer(const td::Slice &data) {
  BWT::byte_buffer answer;
  for (auto i : data) {
    answer.push_back(uint16_t(uint8_t(i)));
  }
  return std::move(answer);
}

td::BufferSlice from_byte_buffer(byte_buffer data) {
  td::BufferSlice ans(data.size());
  auto buffer = get_buffer_slice_data(ans);
  vm::boc_writers::BufferWriter writer{buffer, buffer + data.size()};
  for (auto i : data) {
    writer.store_uint(i, 1);
  }
  return std::move(ans);
}

// Function to perform Burrows-Wheeler Transform
std::pair<byte_buffer, int> bwt(const byte_buffer &input) {
    size_t n = input.size();
    byte_buffer modified_input = input;

    // Append a fictive special symbol (e.g., 0)
    modified_input.push_back(SPECIAL_SYMBOL); // Null byte as special symbol

    size_t modified_n = modified_input.size();

    auto sa = suffixarray(modified_input);

    // Build the BWT output and find the special symbol position
    byte_buffer bwt_output(modified_n); // Exclude the special symbol from the output
    size_t special_symbol_pos = 0;

    for (size_t i = 0; i < modified_n; ++i) {
        data_type value = modified_input[(sa[i] + modified_n - 1) % modified_n];
        if (value == SPECIAL_SYMBOL) { // Check for the special symbol
            special_symbol_pos = i;
        }
        bwt_output[i] = value;
    }

    bwt_output.erase(bwt_output.begin() + special_symbol_pos);

    return {bwt_output, special_symbol_pos};
}

// Function to perform Inverse Burrows-Wheeler Transform
byte_buffer inverse_bwt(byte_buffer bwt_input, size_t special_symbol_pos) {
    bwt_input.insert(bwt_input.begin() + special_symbol_pos, SPECIAL_SYMBOL);
    size_t n = bwt_input.size();
    std::vector<std::pair<data_type, size_t>> sorted_pairs(n);
    
    // Create pairs of (character, original index)
    for (size_t i = 0; i < n; ++i) {
        sorted_pairs[i] = {bwt_input[i], i};
    }

    // Sort pairs by character
    std::sort(sorted_pairs.begin(), sorted_pairs.end());

    // Build the first column of the table
    std::vector<uint8_t> first_col(n);
    for (size_t i = 0; i < n; ++i) {
        first_col[i] = sorted_pairs[i].first;
    }

    // Reconstruct the original string using the last column and the sorted pairs
    byte_buffer original(n);
    size_t current_index = special_symbol_pos;

    for (size_t i = 0; i < n; ++i) {
        original[i] = bwt_input[current_index];
        current_index = sorted_pairs[current_index].second;
    }

    return {original.begin() + 1, original.end()};
}

} // namespace BWT

template<class Writer>
struct BitWriter {
  BitWriter(Writer& _w) : w(_w), bit_value(0), bits(0) {}
  void write_bits(uint64_t value, int bit_size) {
    MSG(log_level::BIT_IO, "Writing bits ", bit_size, '[', value, ']');
    for (int i = 0; i < bit_size; ++i) {
      write_bit(value >> i & 1);
    }
  }
  void write_bit(bool bit) {
    bit_value |= static_cast<uint8_t>(bit) << bits;
    ++bits;
    if (bits == 8) {
      flush_byte();
    }
  }
  void flush_byte() {
    if (bits != 0) {
      MSG(log_level::BIT_IO, "Flushing bits ", bits, '[', uint16_t(bit_value), ']');
      w.store_uint(bit_value, 1); // stores exactly byte
      bits = 0;
      bit_value = 0;
    }
  }
  int position() const {
    return w.position() + bits;
  }
private:
  Writer& w;
  uint8_t bit_value;
  int bits;
};

struct BitReader {
  // BitReader(const td::Slice& _data) : data(_data), ptr(0), bit_index(0) {}
  BitReader(td::Slice _data, int _from_byte) : data(_data), ptr(_from_byte), bit_index(0) {}
  bool read_bit() {
    if (bit_index == 8) {
      flush_byte();
    }
    if (ptr >= data.size()) {
      throw std::range_error("Trying to read more bits, than there is in a slice");
    }
    return data[ptr] >> bit_index++ & 1;
  }
  uint64_t read_bits(int bits) {
    uint64_t ans = 0;
    for (int i = 0; i < bits; ++i) {
      ans |= static_cast<uint64_t>(read_bit()) << i;
    }
    MSG(log_level::BIT_IO, "Read bits ", bits, '[', ans, ']');
    return ans;
  }
  int flush_and_get_ptr() {
    flush_byte();
    return ptr;
  }
  void flush_byte() {
    if (bit_index != 0) {
      MSG(log_level::BIT_IO, "Flush byte");
      ++ptr;
      bit_index = 0;
    }
  }
  td::Slice get_data() const {
    return data;
  }
  int get_ptr() const {
    return ptr;
  }
  int position() const {
    return ptr * 8 + bit_index;
  }
private:
  td::Slice data;
  int ptr;
  int bit_index;
};


namespace huffman {

using distribution_data = std::vector<std::pair<long long, int>>;

struct HuffmanEncoder {
  HuffmanEncoder() {}
  HuffmanEncoder(const distribution_data& data, const std::string name) {
    DBG(log_level::ENCODER_DATA, data);
    set_data(data);
    #ifndef ONLINE_JUDGE
      eval_data(data, name);
    #endif
    build_index();
  }
  void eval_data(const distribution_data& data, const std::string name) {
    long long uncompressed = 0, compressed = 0;
    for (auto [count, value] : data) {
      auto [_, len] = code_len.at(value);
      uncompressed += 8 * count;
      compressed += len * count;
    }
    auto compression_ratio = uncompressed * 1.0 / compressed;
    MSG(log_level::ENCODER_STAT, "Huffman encoder data for ", name, ": ", debug::dbgout(uncompressed, compressed, compression_ratio));
  }
  template<class Writer>
  void write(BitWriter<Writer>& bwriter, int value) const {
    MSG(log_level::NUMBER, "Huffman write ", value);
    auto [code, len] = code_len.at(value);
    auto code_str = std::bitset<64>(code).to_string();
    reverse(code_str.begin(), code_str.end());
    MSG(log_level::NUMBER, "Huffman code ", "0b", code_str.substr(0, len));
    bwriter.write_bits(code, len);
  }
  int read(BitReader& breader) const {
    uint64_t code = 0;
    for (int b = 0; b < 64; ++b) {
      code |= static_cast<uint64_t>(breader.read_bit()) << b;
      // code = (code << 1) + breader.read_bit();
      auto it = code_index.find({code, b + 1});
      if (it != code_index.end()) {
        auto code_str = std::bitset<64>(code).to_string();
        reverse(code_str.begin(), code_str.end());
        MSG(log_level::NUMBER, "Huffman read ", "0b", code_str.substr(b + 1), ' ', it->second);
        return it->second;
      }
    }
    throw std::runtime_error("Read 64 bits but haven't found a match in Huffman Encoder");
  }
  int get_len(int value) const {
    return code_len.at(value).second;
  }
  bool have_value(int value) const {
    return code_len.count(value);
  }
protected:
  void set_data(const distribution_data& data) {
    if (data.size() == 1) {
      code_len[data[0].second] = {0, 0};
      return;
    }
    using vertex = std::pair<long long, std::vector<int>>;
    auto cmp = [&](const vertex& lhs, const vertex& rhs) {
      if (lhs.first != rhs.first) return lhs.first > rhs.first;
      return lhs.second.size() > rhs.second.size();
    };
    std::priority_queue<vertex, std::vector<vertex>, decltype(cmp)> sorted_vertex(cmp);
    for (auto [cnt, value] : data) {
      sorted_vertex.push({cnt, {value}});
    };
    while (sorted_vertex.size() > 1) {
      auto [c1, v1] = sorted_vertex.top();
      sorted_vertex.pop();
      auto [c2, v2] = sorted_vertex.top();
      sorted_vertex.pop();
      for (auto x : v1) add_bit(x, 0);
      for (auto x : v2) add_bit(x, 1);
      v1.insert(v1.end(), v2.begin(), v2.end());
      c1 += c2;
      sorted_vertex.push({c1, v1});
    }
  }
private:
  std::map<int, std::pair<uint64_t, int>> code_len;
  std::map<std::pair<uint64_t, int>, int> code_index;
  void build_index() {
    for (auto [value, c_l] : code_len) {
      auto [code, len] = c_l;
      code_index[{code, len}] = value;
    }
  }
  void add_bit(int value, int bit) {
    auto& [code, len] = code_len[value];
    code = (code << 1) | bit;
    ++len;
    DCHECK(len <= 64);
  }
};


struct HuffmanEncoderWithDefault {
  HuffmanEncoderWithDefault() {}
  HuffmanEncoderWithDefault(distribution_data data, int _special, std::pair<int, int> range, const std::string& name)
  : special(_special)
  , min_value(range.first)
  , extra_bits(std::ceil(std::log2(range.second - range.first))) {
    DBG(log_level::ENCODER_STAT, name, special, min_value, extra_bits);
    for (auto [count, value] : data) {
      DCHECK(value != special);
    }
    data.push_back({0, special});
    encoder = HuffmanEncoder(data, name);
  }

  template<class Writer>
  void write(BitWriter<Writer>& bwriter, int value) const {
    DCHECK(value != special);
    if (encoder.have_value(value)) {
      encoder.write(bwriter, value);
    } else {
      DCHECK(min_value <= value && value < min_value + (1 << extra_bits));
      encoder.write(bwriter, special);
      bwriter.write_bits(value - min_value, extra_bits);
    }
  }
  int read(BitReader& breader) const {
    auto value = encoder.read(breader);
    if (value != special) return value;
    return int(breader.read_bits(extra_bits)) + min_value;
  }
  int get_len(int value) const {
    DCHECK(value != special);
    return encoder.have_value(value) ? encoder.get_len(value) : encoder.get_len(special) + extra_bits;
  }


private:
  HuffmanEncoder encoder;
  int special;
  int min_value;
  int extra_bits;
};

static const std::map<std::string, distribution_data> huffman_data = {
{"prunned_depth",{{1496,2},{1150,9},{784,3},{782,10},{764,23},{730,25},{724,8},{706,22},{696,20},{692,11},{684,21},{684,13},{676,24},{654,12},{650,1},{608,19},{598,14},{596,18},{588,4},{566,26},{536,15},{522,16},{514,27},{502,17},{482,34},{470,28},{452,33},{408,35},{400,6},{398,32},{372,7},{370,38},{354,39},{354,37},{354,5},{346,31},{338,36},{314,29},{310,30},{274,40},{238,41},{152,42},{98,43},{78,100},{76,115},{76,56},{76,0},{70,54},{68,99},{66,114},{64,55},{58,111},{58,52},{52,113},{52,57},{50,112},{48,234},{46,101},{46,98},{46,53},{44,102},{42,116},{38,97},{38,44},{36,446},{36,117},{36,51},{34,120},{34,103},{32,145},{32,118},{32,94},{30,292},{30,191},{28,194},{28,148},{28,144},{28,119},{28,106},{28,95},{28,58},{28,48},{26,289},{26,237},{26,137},{26,121},{24,309},{24,288},{24,200},{24,190},{24,174},{24,146},{24,133},{24,124},{24,110},{24,96},{24,80},{24,64},{24,62},{22,318},{22,290},{22,276},{22,230},{22,182},{22,160},{22,107},{22,89},{22,50},{20,507},{20,357},{20,284},{20,226},{20,213},{20,204},{20,186},{20,167},{20,159},{20,143},{20,132},{20,108},{20,104},{20,88},{20,46},{18,506},{18,462},{18,392},{18,310},{18,275},{18,236},{18,229},{18,227},{18,225},{18,205},{18,173},{18,163},{18,149},{18,126},{18,109},{18,69},{18,49},{16,509},{16,473},{16,329},{16,322},{16,311},{16,273},{16,270},{16,188},{16,158},{16,156},{16,105},{16,93},{16,91},{16,79},{16,66},{16,59},{16,47},{14,441},{14,319},{14,313},{14,304},{14,285},{14,283},{14,255},{14,248},{14,235},{14,209},{14,202},{14,192},{14,189},{14,187},{14,185},{14,165},{14,161},{14,157},{14,151},{14,125},{14,122},{14,76},{14,70},{14,45},{12,508},{12,493},{12,480},{12,478},{12,461},{12,445},{12,421},{12,414},{12,405},{12,384},{12,364},{12,352},{12,351},{12,325},{12,323},{12,317},{12,300},{12,293},{12,287},{12,282},{12,271},{12,267},{12,262},{12,245},{12,239},{12,238},{12,233},{12,232},{12,219},{12,214},{12,211},{12,208},{12,196},{12,195},{12,179},{12,147},{12,141},{12,136},{12,131},{12,85},{12,82},{12,63},{10,519},{10,505},{10,475},{10,474},{10,434},{10,425},{10,423},{10,416},{10,415},{10,390},{10,381},{10,341},{10,331},{10,324},{10,316},{10,314},{10,312},{10,307},{10,298},{10,294},{10,286},{10,277},{10,257},{10,223},{10,210},{10,207},{10,206},{10,193},{10,178},{10,177},{10,168},{10,166},{10,164}}},
{"d1",{{28830,34},{14507,40},{7208,2},{4894,0},{4687,1},{1514,33},{1297,3},{570,35},{97,4},{84,8},{25,10},{24,36},{6,9},{0,255},{0,254},{0,253},{0,252},{0,251},{0,250},{0,249},{0,248},{0,247},{0,246},{0,245},{0,244},{0,243},{0,242},{0,241},{0,240},{0,239},{0,238},{0,237},{0,236},{0,235},{0,234},{0,233},{0,232},{0,231},{0,230},{0,229},{0,228},{0,227},{0,226},{0,225},{0,224},{0,223},{0,222},{0,221},{0,220},{0,219},{0,218},{0,217},{0,216},{0,215},{0,214},{0,213},{0,212},{0,211},{0,210},{0,209},{0,208},{0,207},{0,206},{0,205},{0,204},{0,203},{0,202},{0,201},{0,200},{0,199},{0,198},{0,197},{0,196},{0,195},{0,194},{0,193},{0,192},{0,191},{0,190},{0,189},{0,188},{0,187},{0,186},{0,185},{0,184},{0,183},{0,182},{0,181},{0,180},{0,179},{0,178},{0,177},{0,176},{0,175},{0,174},{0,173},{0,172},{0,171},{0,170},{0,169},{0,168},{0,167},{0,166},{0,165},{0,164},{0,163},{0,162},{0,161},{0,160},{0,159},{0,158},{0,157},{0,156},{0,155},{0,154},{0,153},{0,152},{0,151},{0,150},{0,149},{0,148},{0,147},{0,146},{0,145},{0,144},{0,143},{0,142},{0,141},{0,140},{0,139},{0,138},{0,137},{0,136},{0,135},{0,134},{0,133},{0,132},{0,131},{0,130},{0,129},{0,128},{0,127},{0,126},{0,125},{0,124},{0,123},{0,122},{0,121},{0,120},{0,119},{0,118},{0,117},{0,116},{0,115},{0,114},{0,113},{0,112},{0,111},{0,110},{0,109},{0,108},{0,107},{0,106},{0,105},{0,104},{0,103},{0,102},{0,101},{0,100},{0,99},{0,98},{0,97},{0,96},{0,95},{0,94},{0,93},{0,92},{0,91},{0,90},{0,89},{0,88},{0,87},{0,86},{0,85},{0,84},{0,83},{0,82},{0,81},{0,80},{0,79},{0,78},{0,77},{0,76},{0,75},{0,74},{0,73},{0,72},{0,71},{0,70},{0,69},{0,68},{0,67},{0,66},{0,65},{0,64},{0,63},{0,62},{0,61},{0,60},{0,59},{0,58},{0,57},{0,56},{0,55},{0,54},{0,53},{0,52},{0,51},{0,50},{0,49},{0,48},{0,47},{0,46},{0,45},{0,44},{0,43},{0,42},{0,41},{0,39},{0,38},{0,37},{0,32},{0,31},{0,30},{0,29},{0,28},{0,27},{0,26},{0,25},{0,24},{0,23},{0,22},{0,21},{0,20},{0,19},{0,18},{0,17},{0,16},{0,15},{0,14},{0,13},{0,12},{0,11},{0,7},{0,6},{0,5}}},
{"d2",{{14521,72},{9099,15},{7528,17},{7104,13},{4147,11},{3057,1},{1378,9},{1031,105},{971,130},{909,111},{845,113},{820,19},{777,181},{772,7},{657,177},{558,81},{391,171},{385,67},{348,158},{284,75},{278,89},{277,163},{248,21},{214,161},{212,149},{210,151},{210,23},{210,3},{183,104},{156,175},{154,33},{153,201},{143,153},{143,115},{139,66},{134,152},{134,69},{131,225},{123,135},{120,109},{117,157},{111,156},{111,20},{110,97},{110,73},{109,150},{101,10},{94,91},{91,87},{90,25},{87,147},{86,12},{83,99},{76,16},{74,155},{73,162},{72,112},{66,154},{65,107},{64,80},{60,102},{59,179},{59,169},{55,98},{53,2},{49,121},{49,18},{48,47},{45,5},{43,8},{42,117},{42,65},{41,100},{41,27},{40,160},{38,197},{38,183},{37,229},{36,148},{35,88},{35,68},{34,217},{33,131},{33,37},{32,170},{30,138},{30,14},{29,176},{29,145},{29,143},{29,103},{29,74},{28,173},{28,137},{28,95},{28,94},{27,219},{27,185},{26,178},{25,247},{25,203},{25,172},{24,254},{24,244},{23,222},{23,32},{22,159},{22,55},{22,0},{21,233},{21,174},{21,79},{21,77},{21,26},{21,24},{20,180},{20,110},{20,85},{19,192},{19,167},{19,141},{19,76},{18,235},{17,191},{17,122},{17,71},{17,31},{16,255},{16,83},{15,246},{15,168},{15,132},{15,119},{14,35},{13,251},{13,242},{13,195},{13,165},{13,30},{13,4},{12,230},{12,215},{12,133},{12,114},{12,70},{11,241},{11,227},{11,127},{11,108},{11,22},{10,128},{10,124},{10,123},{10,101},{10,51},{10,34},{10,29},{9,231},{9,182},{9,126},{9,106},{9,38},{8,249},{8,248},{8,220},{8,142},{8,118},{8,64},{8,57},{8,53},{8,28},{7,236},{7,234},{7,208},{7,205},{7,193},{7,129},{7,120},{6,213},{6,202},{6,92},{6,78},{5,252},{5,243},{5,223},{5,212},{5,204},{5,200},{5,166},{5,139},{5,58},{5,56},{5,40},{4,245},{4,221},{4,216},{4,214},{4,209},{4,199},{4,194},{4,189},{4,86},{4,50},{4,46},{4,44},{4,6},{3,226},{3,210},{3,116},{3,96},{3,82},{3,61},{3,60},{3,48},{3,42},{3,39},{2,253},{2,238},{2,224},{2,196},{2,190},{2,187},{2,146},{2,144},{2,140},{2,136},{2,90},{2,84},{2,49},{2,45},{2,36},{1,250},{1,240},{1,232},{1,228},{1,218},{1,211},{1,206},{1,198},{1,184},{1,164},{1,134},{1,93},{1,62},{1,59},{1,54},{1,52},{0,239},{0,237},{0,207},{0,188},{0,186},{0,125},{0,63},{0,43},{0,41}}},
{"ordinary_first_byte",{{26346,0},{1641,190},{1232,32},{1139,12},{1032,114},{846,192},{787,82},{756,64},{711,189},{685,104},{650,1},{611,16},{571,72},{516,185},{511,224},{509,80},{497,201},{475,160},{469,167},{407,223},{391,166},{356,4},{353,191},{334,128},{322,184},{288,102},{274,86},{236,83},{227,136},{203,98},{203,13},{191,115},{189,15},{187,14},{186,67},{182,52},{165,188},{164,66},{156,100},{133,117},{130,118},{126,23},{125,130},{125,108},{115,96},{107,134},{106,135},{104,68},{103,124},{102,119},{97,116},{86,17},{85,221},{85,112},{77,125},{75,255},{75,127},{74,126},{60,8},{59,113},{57,122},{57,120},{53,176},{50,144},{49,84},{49,65},{48,59},{45,194},{44,20},{43,142},{42,5},{41,123},{39,101},{37,121},{35,237},{35,97},{35,49},{31,161},{31,88},{30,212},{30,208},{30,74},{29,155},{27,70},{26,219},{26,3},{25,178},{25,129},{25,48},{23,204},{23,6},{22,248},{22,81},{21,200},{20,175},{20,110},{20,18},{18,207},{18,109},{17,202},{17,62},{17,37},{16,209},{16,187},{16,183},{15,247},{15,173},{15,105},{15,71},{15,19},{14,249},{14,241},{14,195},{14,180},{14,163},{14,69},{13,242},{13,39},{12,169},{12,168},{12,151},{12,143},{12,73},{12,33},{11,228},{11,186},{11,162},{11,99},{10,196},{10,179},{10,95},{10,50},{9,41},{9,22},{8,227},{8,217},{8,147},{8,51},{7,240},{7,222},{7,182},{7,164},{7,106},{7,87},{7,24},{7,7},{6,216},{6,165},{6,140},{6,60},{6,56},{6,10},{5,244},{5,198},{5,181},{5,139},{5,46},{5,44},{5,2},{4,250},{4,234},{4,213},{4,206},{4,205},{4,203},{4,152},{4,138},{4,111},{4,103},{4,85},{4,76},{4,9},{3,253},{3,220},{3,211},{3,210},{3,172},{3,158},{3,157},{3,107},{3,94},{3,89},{3,79},{3,75},{3,53},{3,35},{2,254},{2,252},{2,215},{2,174},{2,159},{2,154},{2,146},{2,141},{2,90},{2,54},{2,47},{1,243},{1,239},{1,235},{1,233},{1,226},{1,177},{1,170},{1,156},{1,149},{1,133},{1,132},{1,91},{1,78},{1,77},{1,63},{1,57},{1,28},{1,21},{0,251},{0,246},{0,245},{0,238},{0,236},{0,232},{0,231},{0,230},{0,229},{0,225},{0,218},{0,214},{0,199},{0,197},{0,193},{0,171},{0,153},{0,150},{0,148},{0,145},{0,137},{0,131},{0,93},{0,92},{0,61},{0,58},{0,55},{0,45},{0,43},{0,42},{0,40},{0,38},{0,36},{0,34},{0,31},{0,30},{0,29},{0,27},{0,26},{0,25},{0,11}}},
{"special_cell_type",{{14507,1},{84,2},{25,4},{6,3},{0,255},{0,254},{0,253},{0,252},{0,251},{0,250},{0,249},{0,248},{0,247},{0,246},{0,245},{0,244},{0,243},{0,242},{0,241},{0,240},{0,239},{0,238},{0,237},{0,236},{0,235},{0,234},{0,233},{0,232},{0,231},{0,230},{0,229},{0,228},{0,227},{0,226},{0,225},{0,224},{0,223},{0,222},{0,221},{0,220},{0,219},{0,218},{0,217},{0,216},{0,215},{0,214},{0,213},{0,212},{0,211},{0,210},{0,209},{0,208},{0,207},{0,206},{0,205},{0,204},{0,203},{0,202},{0,201},{0,200},{0,199},{0,198},{0,197},{0,196},{0,195},{0,194},{0,193},{0,192},{0,191},{0,190},{0,189},{0,188},{0,187},{0,186},{0,185},{0,184},{0,183},{0,182},{0,181},{0,180},{0,179},{0,178},{0,177},{0,176},{0,175},{0,174},{0,173},{0,172},{0,171},{0,170},{0,169},{0,168},{0,167},{0,166},{0,165},{0,164},{0,163},{0,162},{0,161},{0,160},{0,159},{0,158},{0,157},{0,156},{0,155},{0,154},{0,153},{0,152},{0,151},{0,150},{0,149},{0,148},{0,147},{0,146},{0,145},{0,144},{0,143},{0,142},{0,141},{0,140},{0,139},{0,138},{0,137},{0,136},{0,135},{0,134},{0,133},{0,132},{0,131},{0,130},{0,129},{0,128},{0,127},{0,126},{0,125},{0,124},{0,123},{0,122},{0,121},{0,120},{0,119},{0,118},{0,117},{0,116},{0,115},{0,114},{0,113},{0,112},{0,111},{0,110},{0,109},{0,108},{0,107},{0,106},{0,105},{0,104},{0,103},{0,102},{0,101},{0,100},{0,99},{0,98},{0,97},{0,96},{0,95},{0,94},{0,93},{0,92},{0,91},{0,90},{0,89},{0,88},{0,87},{0,86},{0,85},{0,84},{0,83},{0,82},{0,81},{0,80},{0,79},{0,78},{0,77},{0,76},{0,75},{0,74},{0,73},{0,72},{0,71},{0,70},{0,69},{0,68},{0,67},{0,66},{0,65},{0,64},{0,63},{0,62},{0,61},{0,60},{0,59},{0,58},{0,57},{0,56},{0,55},{0,54},{0,53},{0,52},{0,51},{0,50},{0,49},{0,48},{0,47},{0,46},{0,45},{0,44},{0,43},{0,42},{0,41},{0,40},{0,39},{0,38},{0,37},{0,36},{0,35},{0,34},{0,33},{0,32},{0,31},{0,30},{0,29},{0,28},{0,27},{0,26},{0,25},{0,24},{0,23},{0,22},{0,21},{0,20},{0,19},{0,18},{0,17},{0,16},{0,15},{0,14},{0,13},{0,12},{0,11},{0,10},{0,9},{0,8},{0,7},{0,6},{0,5},{0,0}}},
};


static const HuffmanEncoder d1(huffman_data.at("d1"), "d1");
static const HuffmanEncoder d2(huffman_data.at("d2"), "d2");
static const HuffmanEncoder special_cell_type(huffman_data.at("special_cell_type"), "special_cell_type");
static const HuffmanEncoder ordinary_first_byte(huffman_data.at("ordinary_first_byte"), "ordinary_first_byte");
HuffmanEncoder ref_diff;
HuffmanEncoder prunned_depth;

void init_ref_diff(int cell_count) {
  DBG(log_level::ENCODER_STAT, cell_count);
  distribution_data ref_diff_data;
  ref_diff_data.push_back({23000, 0});
  ref_diff_data.push_back({20000, 1});
  ref_diff_data.push_back({11000, 2});
  for (int x = 3; x <= cell_count; ++x) {
    ref_diff_data.push_back({9000 / x, x});
  }
  ref_diff = HuffmanEncoder(ref_diff_data, "ref_diff");
}

void init_prunned_depth() {
  std::vector<int> cnt(1 << 16);
  for (auto [count, value] : huffman_data.at("prunned_depth")) {
    cnt[value] += count;
  }
  distribution_data prunned_depth_data;
  for (int i = 0; i < cnt.size(); ++i) {
    prunned_depth_data.pb({cnt[i], i});
  }
  prunned_depth = HuffmanEncoder(prunned_depth_data, "prunned_depth");
}

} // namespace huffman

namespace compression {

struct SliceTransform {
  virtual td::BufferSlice apply_transform(td::Slice slice) = 0;
  virtual td::BufferSlice revert_transform(td::Slice slice) = 0;
};

enum class FinalCompression {
  ORIGINAL,
  LZ4,
  DEFLATE
};


struct STDCompressor : public SliceTransform {
  enum FinalCompression final_compression;
  STDCompressor(enum FinalCompression _final_compression) : final_compression(_final_compression) {}
  virtual td::BufferSlice apply_transform(td::Slice data) override {
    switch (final_compression) {
      case FinalCompression::ORIGINAL:
        return td::BufferSlice(data);
      case FinalCompression::LZ4:
        return td::lz4_compress(data);
      case FinalCompression::DEFLATE:
        return td::gzencode(data, 2);
      default:
        throw std::invalid_argument("Unknown compression type");
    }
  }
  virtual td::BufferSlice revert_transform(td::Slice data) override {
    switch (final_compression) {
      case FinalCompression::ORIGINAL:
        return td::BufferSlice(data);
      case FinalCompression::LZ4:
        return td::lz4_decompress(data, 2 << 20).move_as_ok();
      case FinalCompression::DEFLATE:
        return td::gzdecode(data);
      default:
        throw std::invalid_argument("Unknown compression type");
    }
  }
};

struct BWTTransform : public SliceTransform {
  virtual td::BufferSlice apply_transform(td::Slice data) override {
    auto [bwt_result, special_symbol_pos] = BWT::bwt(BWT::to_byte_buffer(data));
    BWT::byte_buffer special_sumbol_bytes = {
      special_symbol_pos & 255,
      special_symbol_pos >> 8 & 255,
      special_symbol_pos >> 16 & 255
    };
    bwt_result.insert(bwt_result.begin(), special_sumbol_bytes.begin(), special_sumbol_bytes.end());
    return std::move(BWT::from_byte_buffer(bwt_result));
  }
  virtual td::BufferSlice revert_transform(td::Slice data) override {
    auto ptr = data.ubegin();
    int special_symbol_pos = (int(ptr[0]) << 0) + (int(ptr[1]) << 8) + (int(ptr[2]) << 16);
    data.remove_prefix(3);
    auto inverse_bwt = BWT::inverse_bwt(BWT::to_byte_buffer(data), special_symbol_pos);
    return std::move(BWT::from_byte_buffer(inverse_bwt));
  }
};

} // namespace compression

namespace settings {
  bool use_bwt = false;
  constexpr int PRUNNED_BRANCH_TYPE = 1;
  enum class cell_data_order {
    D1,
    D2,
    SPECIAL_CELL_TYPE,
    CELL_REFS,
    FLUSH_BYTE,
    ORDINARY_FIRST_BYTE,
    PRUNNED_BRANCH_DEPTHS,
    ORDINARY_CELL_DATA,
    PRUNNED_BRANCH_DATA,
    OTHER_SPECIAL_CELLS_DATA,
    SORT_CELLS_BY_META,
  };
  using cell_fields = std::vector<cell_data_order>;
  using cell_field_groups = std::vector<cell_fields>;

  using slice_transforms = std::vector<std::shared_ptr<compression::SliceTransform>>;

  static const std::vector<std::pair<slice_transforms, cell_field_groups>> save_data_order = {
    {
      {
        std::make_shared<compression::STDCompressor>(compression::FinalCompression::DEFLATE),
      },
      {
        {cell_data_order::D1,cell_data_order::D2,cell_data_order::SPECIAL_CELL_TYPE,cell_data_order::ORDINARY_FIRST_BYTE,cell_data_order::FLUSH_BYTE,},
      }
    },
    {
      {
        std::make_shared<compression::STDCompressor>(compression::FinalCompression::DEFLATE),
      },
      {
        {cell_data_order::CELL_REFS,},
        {cell_data_order::SORT_CELLS_BY_META,},
        {cell_data_order::FLUSH_BYTE,cell_data_order::ORDINARY_CELL_DATA,cell_data_order::FLUSH_BYTE},
        {cell_data_order::FLUSH_BYTE, cell_data_order::OTHER_SPECIAL_CELLS_DATA, cell_data_order::FLUSH_BYTE},
        {cell_data_order::PRUNNED_BRANCH_DEPTHS,},
      }
    },
    {
      {
        // Leave this block as is, since it contains totally random data (hashes)
      },
      {
        {cell_data_order::FLUSH_BYTE,cell_data_order::PRUNNED_BRANCH_DATA,cell_data_order::FLUSH_BYTE},
      }
    }
  };

  static constexpr enum compression::FinalCompression final_compression = compression::FinalCompression::DEFLATE;
};

namespace vm {

class CustomBagOfCells {
 public:
  const BagOfCells& get_og() const {
    return *reinterpret_cast<const BagOfCells*>(this);
  }
  BagOfCells& get_og() {
    return *reinterpret_cast<BagOfCells*>(this);
  }
  enum { hash_bytes = vm::Cell::hash_bytes, default_max_roots = 16384 };
  enum Mode { WithIndex = 1, WithCRC32C = 2, WithTopHash = 4, WithIntHashes = 8, WithCacheBits = 16, max = 31 };
  enum { max_cell_whs = 64 };
  using Hash = Cell::Hash;
  struct Info {
    enum : td::uint32 { boc_idx = 0x68ff65f3, boc_idx_crc32c = 0xacc3a728, boc_generic = 0xb5ee9c72 };

    unsigned magic;
    int root_count;
    int cell_count;
    int absent_count;
    int ref_bit_size;
    int offset_bit_size;
    bool valid;
    bool has_index;
    bool has_roots{false};
    bool has_crc32c;
    bool has_cache_bits;
    unsigned long long roots_offset, index_offset, data_offset, data_size, total_size;
    Info() : magic(0), valid(false) {
    }
    void invalidate() {
      valid = false;
    }
    long long parse_serialized_header(BitReader& breader);
  };

  int cell_count{0}, root_count{0}, dangle_count{0}, int_refs{0};
  int int_hashes{0}, top_hashes{0};
  int max_depth{1024};
  Info info;
  unsigned long long data_bytes{0};
  td::HashMap<Hash, int> cells;
  struct CellInfo {
    Ref<DataCell> dc_ref;
    std::array<int, 4> ref_idx;
    unsigned char ref_num;
    unsigned char wt;
    unsigned char hcnt;
    int new_idx;
    bool should_cache{false};
    bool is_root_cell{false};
    CellInfo() : ref_num(0) {
    }
    CellInfo(Ref<DataCell> _dc) : dc_ref(std::move(_dc)), ref_num(0) {
    }
    CellInfo(Ref<DataCell> _dc, int _refs, const std::array<int, 4>& _ref_list)
        : dc_ref(std::move(_dc)), ref_idx(_ref_list), ref_num(static_cast<unsigned char>(_refs)) {
    }
    bool is_special() const {
      return !wt;
    }
  };
  std::vector<CellInfo> cell_list_;
  struct RootInfo {
    RootInfo() = default;
    RootInfo(Ref<Cell> cell, int idx) : cell(std::move(cell)), idx(idx) {
    }
    Ref<Cell> cell;
    int idx{-1};
  };
  std::vector<CellInfo> cell_list_tmp;
  std::vector<RootInfo> roots;
  std::vector<unsigned char> serialized;
  const unsigned char* index_ptr{nullptr};
  const unsigned char* data_ptr{nullptr};
  std::vector<unsigned long long> custom_index;

 public:
  td::Status import_cells() TD_WARN_UNUSED_RESULT;
  CustomBagOfCells() = default;
  std::size_t estimate_serialized_size();
  td::Result<td::BufferSlice> serialize_to_slice();
  td::Result<std::size_t> serialize_to(unsigned char* buffer, std::size_t buff_size);
  template <typename WriterT>
  td::Result<std::size_t> serialize_to_impl(WriterT& writer);

  td::Result<long long> deserialize(const td::Slice& data, int max_roots = default_max_roots);
  td::Result<long long> deserialize(const unsigned char* buffer, std::size_t buff_size,
                                    int max_roots = default_max_roots) {
    return deserialize(td::Slice{buffer, buff_size}, max_roots);
  }
  int get_root_count() const {
    return root_count;
  }
  Ref<Cell> get_root_cell(int idx = 0) const {
    return (idx >= 0 && idx < root_count) ? roots.at(idx).cell : Ref<Cell>{};
  }

  int rv_idx;
  void cells_clear() {
    cell_count = 0;
    int_refs = 0;
    data_bytes = 0;
    cells.clear();
    cell_list_.clear();
  }
  td::uint64 compute_sizes();
};


struct LoadCellData {
  int d1 = -1, d2 = -1, special_cell_type = -1, ordinary_first_byte = -1;
  int uncompressed_data_offset = -1, uncompressed_data_len = -1;
  int prunned_branch_depths_offset = -1;
  std::vector<uint8_t> data;
  std::vector<int> ref_diffs;
  int refs_cnt = -1;
  bool special = false;
  Cell::LevelMask level_mask;
  int full_data_len = -1;
  bool data_with_bits = false;
  td::Result<Ref<DataCell>> custom_create_data_cell(CellBuilder &cb, td::Span<Ref<Cell>> refs) const {
    DCHECK(refs_cnt == (td::int64)refs.size());
    for (int k = 0; k < refs_cnt; k++) {
      cb.store_ref(std::move(refs[k]));
    }
    TRY_RESULT(res, cb.finalize_novm_nothrow(special));
    CHECK(!res.is_null());
    if (res->is_special() != special) {
      return td::Status::Error("is_special mismatch");
    }
    if (res->get_level_mask() != level_mask) {
      return td::Status::Error("level mask mismatch");
    }
    return res;
  }

  td::Result<int> custom_get_bits(td::Slice cell_data) const {
      if (data_with_bits) {
        DCHECK(full_data_len != 0);
        int last = cell_data[full_data_len - 1];
        if (!(last & 0x7f)) {
          return td::Status::Error("overlong encoding");
        }
        return td::narrow_cast<int>((full_data_len - 1) * 8 + 7 - td::count_trailing_zeroes_non_zero32(last));
      } else {
        return td::narrow_cast<int>(full_data_len * 8);
      }
  }

  td::Status init() {
    TRY_STATUS(init_d1());
    return init_d2();
  }

  td::Status init_d1() {
    if (refs_cnt != -1) return td::Status::OK();
    if (d1 == -1) return td::Status::Error("Cell d1 uninitialized");
    DBG(log_level::CELL_META, d1);
    refs_cnt = d1 & 7;
    level_mask = Cell::LevelMask(d1 >> 5);
    special = (d1 & 8) != 0;

    if (refs_cnt > 4) {
      return td::Status::Error("Invalid first byte");
    }

    DBG(log_level::CELL_META, refs_cnt, level_mask.get_mask(), special);

    return td::Status::OK();
  }

  td::Status init_d2() {
    if (full_data_len != -1) return td::Status::OK();
    if (d2 == -1) return td::Status::Error("Cell d2 uninitialized");
    DBG(log_level::CELL_META, d2);

    full_data_len = (d2 >> 1) + (d2 & 1);
    data_with_bits = (d2 & 1) != 0;

    DBG(log_level::CELL_META, full_data_len, data_with_bits);

    return td::Status::OK();
  }

  td::Status init_data() {
    if (data.capacity() > 0) return td::Status::OK();
    TRY_STATUS(init());
    data.resize(256); // Initializes data.data() even if full_data_len = 0
    data.resize(full_data_len);
    return td::Status::OK();
  }

  td::Status init_offsets() {
    if (uncompressed_data_offset != -1) return td::Status::OK();
    TRY_STATUS(init_data());
    if (special) {
      if (special_cell_type == -1) {
        return td::Status::Error("Can't initialize data when special cell type is not set");
      }
      if (special_cell_type == settings::PRUNNED_BRANCH_TYPE) {
        // TODO: Check that data[1] = 1 on more random tests,
        // but seems to be fine on 100 system tests
        DCHECK(full_data_len >= 2);
        data[1] = 1;
        uncompressed_data_offset = 2;
        uncompressed_data_len = 32 * prunned_branches_count();
        prunned_branch_depths_offset = uncompressed_data_offset + uncompressed_data_len;
      } else {
        uncompressed_data_offset = 1;
        uncompressed_data_len = full_data_len - uncompressed_data_offset;
      }
    } else if (full_data_len > 0) {
      uncompressed_data_offset = 1;
      uncompressed_data_len = full_data_len - 1;
    }
    return td::Status::OK();
  }

  td::Status set_special_cell_type(int type) {
    TRY_STATUS(init_data());
    if (!special) return td::Status::OK();
    special_cell_type = type;
    DBG(log_level::CELL_META, special_cell_type);
    DCHECK(full_data_len >= 1);
    data[0] = special_cell_type;
    return td::Status::OK();
  }

  td::Status set_ordinary_first_byte(int type) {
    TRY_STATUS(init_data());
    if (special || full_data_len == 0) return td::Status::OK();
    ordinary_first_byte = type;
    DBG(log_level::CELL_META, ordinary_first_byte);
    DCHECK(full_data_len >= 1);
    data[0] = ordinary_first_byte;
    return td::Status::OK();
  }

  int prunned_branches_count() const {
    DCHECK(special && special_cell_type == 1);
    return (full_data_len - 2) / 34;
  }

  template<class Writer>
  td::Status store_prunned_branch_depths(BitWriter<Writer>& bwriter) {
    TRY_STATUS(init_d1());
    if (!special) return td::Status::OK();
    TRY_STATUS(init_offsets());
    if (special_cell_type == -1) return td::Status::Error("Special cell type uninitialized");
    if (special_cell_type != settings::PRUNNED_BRANCH_TYPE) return td::Status::OK();
    for (int x = 0; x < prunned_branches_count(); ++x) {
      int id = prunned_branch_depths_offset + 2 * x;
      int val = (uint16_t(data[id]) << 8) | uint16_t(data[id + 1]);
      add_int("prunned_depth", val);
      // bwriter.write_bits(buf[id], 8);
      // bwriter.write_bits(buf[id + 1], 8);
      huffman::prunned_depth.write(bwriter, val);
    }
    return td::Status::OK();
  }
  td::Status load_prunned_branch_depths(BitReader& breader) {
    TRY_STATUS(init_d1());
    if (!special) return td::Status::OK();
    if (special_cell_type == -1) return td::Status::Error("Special cell type uninitialized");
    if (special_cell_type != settings::PRUNNED_BRANCH_TYPE) return td::Status::OK();
    TRY_STATUS(init_offsets());
    for (int i = 0; i < prunned_branches_count(); ++i) {
      int id = prunned_branch_depths_offset + 2 * i;
      // data[id] = breader.read_bits(8);
      // data[id + 1] = breader.read_bits(8);
      int val = huffman::prunned_depth.read(breader);
      data[id] = val >> 8;
      data[id + 1] = val & 255;
    }
    return td::Status::OK();
  }

  template<class Writer>
  td::Status store_ref_diffs(BitWriter<Writer>& bwriter) {
    for (auto ref_diff : ref_diffs) {
      add_int("ref_diff", ref_diff);
      huffman::ref_diff.write(bwriter, ref_diff);
      // store_ref(ref_diff);
    }
    DBG(log_level::LOG_LEVEL::SKIP, ref_diffs);
    return td::Status::OK();
  }
  td::Status load_ref_diffs(BitReader& breader) {
    DBG(log_level::CELL_META, refs_cnt);
    ref_diffs.resize(refs_cnt);
    for (auto &diff : ref_diffs) {
      // diff = breader.read_bits(info.ref_bit_size);
      diff = huffman::ref_diff.read(breader);
    }
    return td::Status::OK();
  }

  template<class Writer>
  td::Status store_cell_uncompressed_data(BitWriter<Writer>& bwriter) {
    TRY_STATUS(init_offsets());
    for (int i = uncompressed_data_offset; i < uncompressed_data_offset + uncompressed_data_len; ++i) {
      MSG(log_level::LOG_LEVEL::BYTE, "Writing data ", uint16_t(data[i]));
      bwriter.write_bits(data[i], 8);
    }
    return td::Status::OK();
  }
  td::Status load_cell_uncompressed_data(BitReader& breader) {
    TRY_STATUS(init_offsets());
    for (int i = uncompressed_data_offset; i < uncompressed_data_offset + uncompressed_data_len; ++i) {
      MSG(log_level::LOG_LEVEL::BYTE, "Reading data ", uint16_t(data[i]));
      data[i] = breader.read_bits(8);
    }
    return td::Status::OK();
  }

  // int compare_prunned_branches(const LoadCellData& other) const {
  //   if (special_cell_type != 1 || other.special_cell_type != 1) return 0;
  //   int count = prunned_branches_count();
  //   int other_count = prunned_branches_count();
  //   if (count != other_count) {
  //     return count < other_count ? -1 : 1;
  //   }
  //   return memcmp(data.data() + prunned_branch_depths_offset,
  //                 other.data.data() + other.prunned_branch_depths_offset,
  //                 count * 2);
  // }

  // return <0 if a < b, >0 if a > b and 0 if a == b
  int compare_meta(const LoadCellData& other) const {
    if (level_mask.get_mask() != other.level_mask.get_mask()) {
      return level_mask.get_mask() < other.level_mask.get_mask() ? -1 : 1;
    }
    if (refs_cnt != other.refs_cnt) {
      return refs_cnt < other.refs_cnt ? -1 : 1;
    }
    if (d2 != other.d2) {
      return d2 < other.d2 ? -1 : 1;
    }
    if (ordinary_first_byte != other.ordinary_first_byte) {
      return ordinary_first_byte < other.ordinary_first_byte ? -1 : 1;
    }
    if (special_cell_type != other.special_cell_type) {
      return special_cell_type < other.special_cell_type ? -1 : 1;
    }
    // auto check_prunned_branches = compare_prunned_branches(other);
    // if (check_prunned_branches != 0) return check_prunned_branches;
    // if (ref_diffs != other.ref_diffs) {
    //   return ref_diffs < other.ref_diffs ? -1 : 1;
    // }
    return 0;
  }
};

long long CustomBagOfCells::Info::parse_serialized_header(BitReader& breader) {
  invalidate();
  root_count = cell_count = -1;
  index_offset = data_offset = total_size = 0;
  cell_count = breader.read_bits(16);
  if (cell_count <= 0) {
    cell_count = -1;
    return 0;
  }
  DBG(log_level::COMPRESSION_META, cell_count);
  huffman::init_ref_diff(cell_count);
  root_count = 1;
  if (root_count <= 0) {
    root_count = -1;
    return 0;
  }
  has_roots = true;
  valid = true;

  return 1;
}

template <typename WriterT>
td::Result<std::size_t> CustomBagOfCells::serialize_to_impl(WriterT& writer) {
  BitWriter bwriter(writer);
  MSG(log_level::COMPRESSION_META, "Running custom serialize impl");

  bwriter.write_bits(cell_count, 16);
  huffman::init_ref_diff(cell_count);
  DCHECK((unsigned)cell_count == cell_list_.size());

  std::vector<std::vector<uint8_t>> serialized_cells(cell_count, std::vector<uint8_t>(256));
  std::vector<LoadCellData> cell_info(cell_count);

  for (int i = cell_count - 1; i >= 0; --i) {
    int idx = cell_count - 1 - i;
    const auto& dc_info = cell_list_[idx];
    const Ref<DataCell>& dc = dc_info.dc_ref;
    MSG(log_level::CELL_META, "Serializing cell with idx ", i, " with refnum ", int(dc_info.ref_num));
    int s = dc->serialize(serialized_cells[i].data(), 256);
    MSG(log_level::CELL_META, "Cell serialized size = ", s);
    serialized_cells[i].resize(s);
    auto* buf = serialized_cells[i].data();
    cell_info[i].d1 = uint16_t(buf[0]);
    cell_info[i].d2 = uint16_t(buf[1]);
    TRY_STATUS(cell_info[i].init_data());
    if (cell_info[i].special) {
      cell_info[i].set_special_cell_type(uint32_t(buf[2]));
    } else if (s >= 2) {
      cell_info[i].set_ordinary_first_byte(uint32_t(buf[2]));
    }
    for (int x = 2; x < s; ++x) {
      cell_info[i].data[x - 2] = buf[x];
    }
    TRY_STATUS(cell_info[i].init_offsets());
    if (!cell_info[i].special && cell_info[i].full_data_len >= 2) {
      add_int("two_bytes", (uint16_t(buf[2]) << 8) + uint16_t(buf[3]));
    }
    auto get_cell_ref_diffs = [&]() {
      std::vector<int> ref_diffs;
      for (unsigned j = 0; j < dc_info.ref_num; ++j) {
        int k = cell_count - 1 - dc_info.ref_idx[j];
        MSG(log_level::CELL_META, "Link from ", i, " to ", k);
        DCHECK(k > i && k < cell_count);
        int ref_diff = k - i - 1;
        ref_diffs.push_back(ref_diff);
      }
      return ref_diffs;
    };
    cell_info[i].ref_diffs = get_cell_ref_diffs();
  }

  std::vector<int> cell_order(cell_count);
  std::iota(cell_order.begin(), cell_order.end(), 0);
  std::reverse(cell_order.begin(), cell_order.end());

  static const int buff_size = 2 << 20;
  td::BufferSlice buff_slice(buff_size);


  for (const auto& [transforms, stored_groups_of_fields] : settings::save_data_order) {
    auto* buff = get_buffer_slice_data(buff_slice);
    boc_writers::BufferWriter buffer_writer{buff, buff + buff_size};
    BitWriter buffer_bwriter(buffer_writer);

    for (auto stored_fields : stored_groups_of_fields) {
      for (auto i : cell_order) {
        int idx = cell_count - 1 - i;
        MSG(log_level::CELL_META, "Saving cell with idx ", i, " with refnum ", int(cell_list_[idx].ref_num));
        auto start_position = buffer_bwriter.position();
        const auto& dc_info = cell_list_[idx];
        const Ref<DataCell>& dc = dc_info.dc_ref;
        for (auto mode : stored_fields) {
          switch (mode) {
            case settings::cell_data_order::SORT_CELLS_BY_META: {
              MSG(log_level::COMPRESSION_META, "Sorting cells by meta info");
              std::sort(cell_order.begin(), cell_order.end(), [&](int a, int b) {
                auto check = cell_info[a].compare_meta(cell_info[b]);
                if (check != 0) return check < 0;
                return a > b;
              });
              goto next_save_data_order;
            }
            case settings::cell_data_order::D1: {
              auto d1 = cell_info[i].d1;
              DBG(log_level::CELL_META, d1);
              add_char("d1", d1);
              huffman::d1.write(buffer_bwriter, d1);
              break;
            }
            case settings::cell_data_order::D2: {
              auto d2 = cell_info[i].d2;
              DBG(log_level::CELL_META, d2);
              add_char("d2", d2);
              huffman::d2.write(buffer_bwriter, d2);
              break;
            }
            case settings::cell_data_order::SPECIAL_CELL_TYPE: {
              if (cell_info[i].special) {
                auto special_type = cell_info[i].special_cell_type;
                add_char("special_cell_type", special_type);
                huffman::special_cell_type.write(buffer_bwriter, special_type);
              }
              break;
            }
            case settings::cell_data_order::ORDINARY_FIRST_BYTE: {
              if (!cell_info[i].special && cell_info[i].full_data_len > 0) {
                auto first_byte = cell_info[i].ordinary_first_byte;
                add_char("ordinary_first_byte", first_byte);
                huffman::ordinary_first_byte.write(buffer_bwriter, first_byte);
              }
              break;
            }
            case settings::cell_data_order::FLUSH_BYTE: {
              MSG(log_level::BIT_IO, "Custom byte flush");
              buffer_bwriter.flush_byte();
              break;
            }
            case settings::cell_data_order::ORDINARY_CELL_DATA: {
              if (cell_info[i].special) continue; 
              cell_info[i].store_cell_uncompressed_data(buffer_bwriter);
              break;
            }
            case settings::cell_data_order::PRUNNED_BRANCH_DATA: {
              if (!cell_info[i].special || cell_info[i].special_cell_type != settings::PRUNNED_BRANCH_TYPE) continue;
              cell_info[i].store_cell_uncompressed_data(buffer_bwriter);
              break;
            }
            case settings::cell_data_order::OTHER_SPECIAL_CELLS_DATA: {
              if (!cell_info[i].special || cell_info[i].special_cell_type == settings::PRUNNED_BRANCH_TYPE) continue;
              cell_info[i].store_cell_uncompressed_data(buffer_bwriter);
              break;
            }
            case settings::cell_data_order::PRUNNED_BRANCH_DEPTHS: {
              TRY_STATUS(cell_info[i].store_prunned_branch_depths(buffer_bwriter));
              break;
            }
            case settings::cell_data_order::CELL_REFS: {
              DCHECK(dc->size_refs() == dc_info.ref_num);
              TRY_STATUS(cell_info[i].store_ref_diffs(buffer_bwriter));
              break;
            }
            default:
              throw std::logic_error("Not implemented data saving");
              break;
          }
        }
        auto end_position = buffer_bwriter.position();
        MSG(log_level::CELL_META, "Cell position ", start_position, ' ', end_position);
      }
      next_save_data_order:;
    }
    buffer_bwriter.flush_byte();
    auto written_bytes = buffer_writer.position();
    if (written_bytes == 0) continue;
    td::BufferSlice saved_data = td::BufferSlice(reinterpret_cast<char*>(get_buffer_slice_data(buff_slice)), written_bytes);
    if (log_level::check_log_level(log_level::LOG_LEVEL::BYTE)) {
      msg("Saving slice of size ", saved_data.size());
      for (int i = 0; i < saved_data.size(); ++i) {
        std::cerr << uint16_t(uint8_t(saved_data[i])) << ' ';
      }
      std::cerr << '\n';
    }

    for (const auto& transform : transforms) {
      saved_data = transform->apply_transform(saved_data);
    }

    MSG(log_level::COMPRESSION_META, "Written bytes = ", written_bytes, ", transformed = ", saved_data.size(), ", CR = ", written_bytes * 1.0 / saved_data.size());

    bwriter.write_bits(saved_data.size(), 24);
    for (int i = 0; i < saved_data.size(); ++i) {
      bwriter.write_bits(uint8_t(saved_data[i]), 8);
    }

  }


  bwriter.flush_byte(); // It's important to write the last byte, otherwise it will stay in the buffer
  writer.chk();
  DBG(log_level::COMPRESSION_META, writer.position());
  return writer.position();
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
td::uint64 CustomBagOfCells::compute_sizes() {
  td::uint64 data_bytes_adj = cell_count * 3 + data_bytes + (unsigned long long)int_refs * 4;
  return data_bytes_adj;
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
std::size_t CustomBagOfCells::estimate_serialized_size() {
  auto data_bytes_adj = compute_sizes();
  if (!data_bytes_adj) {
    info.invalidate();
    return 0;
  }
  info.valid = true;
  info.cell_count = cell_count;
  // TODO: Maybe make this constant bigger for safety
  info.total_size = 10 + data_bytes_adj;
  auto res = td::narrow_cast_safe<size_t>(info.total_size);
  if (res.is_error()) {
    return 0;
  }
  return res.ok() + 1;
}

td::Result<std::size_t> CustomBagOfCells::serialize_to(unsigned char* buffer, std::size_t buff_size) {
  boc_writers::BufferWriter writer{buffer, buffer + buff_size};
  return serialize_to_impl(writer);
}

td::Result<td::BufferSlice> CustomBagOfCells::serialize_to_slice() {
  std::size_t size_est = estimate_serialized_size();
  if (!size_est) {
    return td::Status::Error("no cells to serialize to this bag of cells");
  }
  int buff_size = 4 << 20;
  td::BufferSlice res(buff_size);
  TRY_RESULT(size, serialize_to(get_buffer_slice_data(res), res.size()));
  MSG(log_level::COMPRESSION_META, "Expected size = ", size_est, ", Real size = ", size);
  if (size <= res.size()) {
    res.truncate(size);
    return std::move(res);
  } else {
    return td::Status::Error("error while serializing a bag of cells: actual serialized size differs from estimated");
  }
}

td::Result<long long> CustomBagOfCells::deserialize(const td::Slice& data, int max_roots) {
  MSG(log_level::COMPRESSION_META, "Running custom deserialize impl");
  get_og().clear();
  BitReader breader(data, 0);
  long long start_offset = info.parse_serialized_header(breader);
  if (start_offset == 0) {
    return td::Status::Error(PSLICE() << "cannot deserialize bag-of-cells: invalid header, error " << start_offset);
  }
  if (start_offset < 0) {
    return start_offset;
  }

  if (info.root_count > max_roots) {
    return td::Status::Error("Bag-of-cells has more root cells than expected");
  }

  cell_count = info.cell_count;
  roots.clear();
  roots.resize(info.root_count);
  roots[0].idx = info.cell_count - 1;
  std::vector<Ref<DataCell>> cell_list;
  cell_list.reserve(cell_count);
  std::vector<LoadCellData> cell_data(cell_count);

  std::vector<int> cell_order(cell_count);
  std::iota(cell_order.begin(), cell_order.end(), 0);

  td::BufferSlice buffer;

  for (const auto& [transforms, stored_groups_of_fields] : settings::save_data_order) {
    int data_len = breader.read_bits(24);

    buffer = td::BufferSlice(data_len);

    for (int i = 0; i < data_len; ++i) {
      buffer.as_slice()[i] = breader.read_bits(8);
    }

    for (int i = int(transforms.size()) - 1; i >= 0; --i) {
      buffer = transforms[i]->revert_transform(buffer.as_slice());
    }

    if (log_level::check_log_level(log_level::LOG_LEVEL::BYTE)) {
      msg("Loaded slice of size ", buffer.size());
      for (int i = 0; i < buffer.size(); ++i) {
        std::cerr << uint16_t(uint8_t(buffer[i])) << ' ';
      }
      std::cerr << '\n';
    }

    BitReader buffer_breader(buffer, 0);

    for (auto stored_fields : stored_groups_of_fields) {
      for (auto i : cell_order) {
        int idx = cell_count - 1 - i;
        MSG(log_level::CELL_META, "Deserializing cell with idx ", idx);
        auto& cell_info = cell_data[i];
        for (auto mode : stored_fields) {
          switch (mode) {
            case settings::cell_data_order::SORT_CELLS_BY_META: {
              std::sort(cell_order.begin(), cell_order.end(), [&](int a, int b) {
                auto check = cell_data[a].compare_meta(cell_data[b]);
                if (check != 0) return check == -1;
                return a < b;
              });
              goto next_load_data_order;
            }
            case settings::cell_data_order::D1: {
              cell_info.d1 = huffman::d1.read(buffer_breader);
              break;
            }
            case settings::cell_data_order::D2: {
              cell_info.d2 = huffman::d2.read(buffer_breader);
              break;
            }
            case settings::cell_data_order::SPECIAL_CELL_TYPE: {
              TRY_STATUS(cell_info.init_d1());
              if (cell_info.special) {
                auto special_cell_type = huffman::special_cell_type.read(buffer_breader);
                cell_info.set_special_cell_type(special_cell_type);
              }
              break;
            }
            case settings::cell_data_order::ORDINARY_FIRST_BYTE: {
              TRY_STATUS(cell_info.init());
              if (!cell_info.special && cell_info.full_data_len > 0) {
                int ordinary_first_byte = huffman::ordinary_first_byte.read(buffer_breader);
                cell_info.set_ordinary_first_byte(ordinary_first_byte);
              }
              break;
            }
            case settings::cell_data_order::FLUSH_BYTE: {
              MSG(log_level::BIT_IO, "Custom flush byte");
              buffer_breader.flush_byte();
              break;
            }
            case settings::cell_data_order::ORDINARY_CELL_DATA: {
              TRY_STATUS(cell_info.init_d1());
              if (!cell_info.special) {
                TRY_STATUS(cell_info.load_cell_uncompressed_data(buffer_breader));
              }
              break;
            }
            case settings::cell_data_order::PRUNNED_BRANCH_DATA: {
              TRY_STATUS(cell_info.init_d1());
              if (cell_info.special && cell_info.special_cell_type == settings::PRUNNED_BRANCH_TYPE) {
                TRY_STATUS(cell_info.load_cell_uncompressed_data(buffer_breader));
              }
              break;
            }
            case settings::cell_data_order::OTHER_SPECIAL_CELLS_DATA: {
              TRY_STATUS(cell_info.init_d1());
              if (cell_info.special && cell_info.special_cell_type != settings::PRUNNED_BRANCH_TYPE) {
                TRY_STATUS(cell_info.load_cell_uncompressed_data(buffer_breader));
              }
              break;
            }
            case settings::cell_data_order::CELL_REFS: {
              TRY_STATUS(cell_info.init_d1());
              TRY_STATUS(cell_info.load_ref_diffs(buffer_breader));
              break;
            }
            case settings::cell_data_order::PRUNNED_BRANCH_DEPTHS: {
              TRY_STATUS(cell_info.load_prunned_branch_depths(buffer_breader));
              break;
            }

            default:
              throw std::logic_error("Loading data not implemented");
          }
        }
      }
      next_load_data_order:;
    }
  }
  for (int i = 0; i < cell_count; i++) {
    // reconstruct cell with index cell_count - 1 - i
    int idx = cell_count - 1 - i;
    MSG(log_level::CELL_META, "Loading cell with idx ", idx);
    auto& cell_info = cell_data[i];
    cell_info.set_special_cell_type(cell_info.special_cell_type);
    cell_info.set_ordinary_first_byte(cell_info.ordinary_first_byte);

    CellBuilder cb;

    td::Slice cell_slice(cell_info.data.data(), cell_info.data.size());
    TRY_RESULT(bits, cell_info.custom_get_bits(cell_slice));
    MSG(log_level::CELL_META, "Cell bits size = ", bits);
    cb.store_bits(cell_slice.data(), bits);

    std::array<td::Ref<Cell>, 4> refs_buf;
    auto refs = td::MutableSpan<td::Ref<Cell>>(refs_buf).substr(0, cell_info.refs_cnt);
    for (int k = 0; k < cell_info.refs_cnt; k++) {
      // int ref_diff = huffman::ref_diff.read(breader);
      int ref_diff = cell_info.ref_diffs[k];
      int ref_idx = idx + 1 + ref_diff;
      if (ref_idx <= idx) {
        return td::Status::Error(PSLICE() << "bag-of-cells error: reference #" << k << " of cell #" << idx
                                          << " is to cell #" << ref_idx << " with smaller index");
      }
      if (ref_idx >= cell_count) {
        return td::Status::Error(PSLICE() << "bag-of-cells error: reference #" << k << " of cell #" << idx
                                          << " is to non-existent cell #" << ref_idx << ", only " << cell_count
                                          << " cells are defined");
      }
      refs[k] = cell_list[cell_count - ref_idx - 1];
    }
    auto r_cell = cell_info.custom_create_data_cell(cb, refs);
    cell_list.push_back(r_cell.move_as_ok());
    DCHECK(cell_list.back().not_null());
  }
  auto end_offset = breader.flush_and_get_ptr();
  root_count = info.root_count;
  for (auto& root_info : roots) {
    root_info.cell = cell_list[root_info.idx];
  }
  cell_list.clear();
  return end_offset;
}

td::Result<td::BufferSlice> custom_boc_serialize(Ref<Cell> root) {
  if (root.is_null()) {
    return td::Status::Error("cannot serialize a null cell reference into a bag of cells");
  }
  BagOfCells boc;
  boc.add_root(std::move(root));
  auto res = boc.import_cells();
  if (res.is_error()) {
    return res.move_as_error();
  }
  auto custom_boc = *reinterpret_cast<CustomBagOfCells*>(&boc);
  return custom_boc.serialize_to_slice();
}
td::Result<Ref<Cell>> custom_boc_deserialize(td::Slice data, bool can_be_empty = false, bool allow_nonzero_level = false) {
  if (data.empty() && can_be_empty) {
    return Ref<Cell>();
  }
  CustomBagOfCells custom_boc;
  auto res = custom_boc.deserialize(data, 1);
  if (res.is_error()) {
    return res.move_as_error();
  }
  auto boc = *reinterpret_cast<BagOfCells*>(&custom_boc);
  if (boc.get_root_count() != 1) {
    return td::Status::Error("bag of cells is expected to have exactly one root");
  }
  auto root = boc.get_root_cell();
  if (root.is_null()) {
    return td::Status::Error("bag of cells has null root cell (?)");
  }
  if (!allow_nonzero_level && root->get_level() != 0) {
    return td::Status::Error("bag of cells has a root with non-zero level");
  }
  return std::move(root);
}

} // namespace vm

td::BufferSlice compress(td::Slice data) {
  td::Ref<vm::Cell> root = vm::std_boc_deserialize(data).move_as_ok();
  td::BufferSlice serialized = vm::custom_boc_serialize(root).move_as_ok();
  return serialized;
}

td::BufferSlice decompress(td::Slice data) {
  vm::Ref<vm::Cell> root = vm::custom_boc_deserialize(data).move_as_ok();
  return vm::std_boc_serialize(root, 31).move_as_ok();
}

int main() {
  huffman::init_prunned_depth();
  std::string mode;
  std::cin >> mode;
  CHECK(mode == "compress" || mode == "decompress");

  std::string base64_data;
  std::cin >> base64_data;
  CHECK(!base64_data.empty());

  td::BufferSlice data(td::base64_decode(base64_data).move_as_ok());

  if (mode == "compress") {
    data = compress(data);
  } else {
    data = decompress(data);
  }

  std::cout << td::base64_encode(data) << std::endl;

  #ifndef ONLINE_JUDGE
    std::ignore = freopen("res/raw_huffman_data.txt", "a", stdout);

    for (auto &[name, data] : byte_cnt) {
      std::vector<std::pair<int, int>> st_data(data.begin(), data.end());
      std::sort(st_data.begin(), st_data.end(), [&](auto a, auto b) {
        return a.second > b.second;
      });
      long long total = 0;
      huffman::distribution_data distr; 
      for (auto [value, count] : st_data) {
        std::cout << name << ' ' << value << ' ' << count << '\n';
        total += count;
        distr.push_back({count, value});
      }
      MSG(log_level::ENCODER_STAT, "Different count for ", name, " = ", data.size(), " Total count = ", total, " ", data.size() * 1.0 / total);
      huffman::HuffmanEncoder encoder(distr, name);
    }
  #endif
}

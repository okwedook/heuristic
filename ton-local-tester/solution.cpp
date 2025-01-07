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

  static constexpr auto ENCODER_BUILD = LOG_LEVEL::BYTE;
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

namespace settings {
  enum class CELL_DATA_ORDER {
    d1,
    d2,
    cell_type,
    cell_data,
    cell_refs,
    flush_byte
  };
  static const std::vector<enum CELL_DATA_ORDER> save_data_order = {
    CELL_DATA_ORDER::d1,
    CELL_DATA_ORDER::d2,
    CELL_DATA_ORDER::cell_type,
    CELL_DATA_ORDER::flush_byte,
    CELL_DATA_ORDER::cell_data,
    CELL_DATA_ORDER::cell_refs,
  };
};

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
  auto buffer = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(ans.data()));
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
    MSG(log_level::BIT_IO, "Writing bits ", value, ' ', bit_size);
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
      MSG(log_level::BIT_IO, "Writing bits ", uint16_t(bit_value), ' ', bits);
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
  BitReader(const td::Slice& _data, int _from_byte) : data(_data), ptr(_from_byte), bit_index(0) {}
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
    MSG(log_level::BIT_IO, "Read bits ", ans, ' ', bits);
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
  const td::Slice& get_data() const {
    return data;
  }
  int get_ptr() const {
    return ptr;
  }
  int position() const {
    return ptr * 8 + bit_index;
  }
private:
  const td::Slice& data;
  int ptr;
  int bit_index;
};


namespace huffman {

using distribution_data = std::vector<std::pair<long long, int>>;

static const std::map<std::string, distribution_data> huffman_data = {
{"byte_depth",{{748,2},{575,9},{392,3},{391,10},{382,23},{365,25},{362,8},{353,22},{348,20},{346,11},{342,21},{342,13},{338,24},{327,12},{325,1},{304,19},{299,14},{298,18},{294,4},{283,26},{268,15},{261,16},{257,27},{251,17},{241,34},{235,28},{226,33},{204,35},{200,6},{199,32},{186,7},{185,38},{177,39},{177,37},{177,5},{173,31},{169,36},{157,29},{155,30},{137,40},{119,41},{76,42},{49,43},{39,100},{38,115},{38,56},{38,0},{35,54},{34,99},{33,114},{32,55},{29,111},{29,52},{26,113},{26,57},{25,112},{24,234},{23,101},{23,98},{23,53},{22,102},{21,116},{19,97},{19,44},{18,446},{18,117},{18,51},{17,120},{17,103},{16,145},{16,118},{16,94},{15,292},{15,191},{14,194},{14,148},{14,144},{14,119},{14,106},{14,95},{14,58},{14,48},{13,289},{13,237},{13,137},{13,121},{12,309},{12,288},{12,200},{12,190},{12,174},{12,146},{12,133},{12,124},{12,110},{12,96},{12,80},{12,64},{12,62},{11,318},{11,290},{11,276},{11,230},{11,182},{11,160},{11,107},{11,89},{11,50},{10,507},{10,357},{10,284},{10,226},{10,213},{10,204},{10,186},{10,167},{10,159},{10,143},{10,132},{10,108},{10,104},{10,88},{10,46},{9,506},{9,462},{9,392},{9,310},{9,275},{9,236},{9,229},{9,227},{9,225},{9,205},{9,173},{9,163},{9,149},{9,126},{9,109},{9,69},{9,49},{8,509},{8,473},{8,329},{8,322},{8,311},{8,273},{8,270},{8,188},{8,158},{8,156},{8,105},{8,93},{8,91},{8,79},{8,66},{8,59},{8,47},{7,441},{7,319},{7,313},{7,304},{7,285},{7,283},{7,255},{7,248},{7,235},{7,209},{7,202},{7,192},{7,189},{7,187},{7,185},{7,165},{7,161},{7,157},{7,151},{7,125},{7,122},{7,76},{7,70},{7,45},{6,508},{6,493},{6,480},{6,478},{6,461},{6,445},{6,421},{6,414},{6,405},{6,384},{6,364},{6,352},{6,351},{6,325},{6,323},{6,317},{6,300},{6,293},{6,287},{6,282},{6,271},{6,267},{6,262},{6,245},{6,239},{6,238},{6,233},{6,232},{6,219},{6,214},{6,211},{6,208},{6,196},{6,195},{6,179},{6,147},{6,141},{6,136},{6,131},{6,85},{6,82},{6,63},{5,519},{5,505},{5,475},{5,474},{5,434},{5,425},{5,423},{5,416},{5,415},{5,390},{5,381},{5,341},{5,331},{5,324},{5,316},{5,314},{5,312},{5,307},{5,298},{5,294},{5,286},{5,277},{5,257},{5,223},{5,210},{5,207},{5,206},{5,193},{5,178},{5,177},{5,168},{5,166},{5,164}}},
{"cell_data",{{110109,0},{25151,1},{12349,2},{12017,192},{11409,64},{11198,160},{10487,8},{9617,40},{9460,128},{9290,232},{8873,200},{8859,136},{8673,11},{8526,104},{8438,168},{8354,72},{8314,32},{8238,224},{7769,16},{7716,6},{7698,4},{7572,3},{7544,23},{7183,48},{7020,24},{6983,96},{6940,80},{6826,5},{6775,208},{6769,193},{6741,22},{6702,112},{6630,9},{6624,10},{6583,144},{6582,12},{6559,132},{6559,20},{6514,51},{6424,17},{6323,194},{6262,163},{6239,13},{6238,56},{6221,49},{6166,130},{6158,52},{6136,33},{6112,93},{6083,67},{6072,226},{6071,255},{6050,50},{6044,203},{6041,161},{6040,54},{6039,129},{6027,170},{6024,97},{6006,206},{5997,116},{5972,76},{5936,19},{5935,176},{5922,250},{5889,162},{5889,35},{5888,14},{5869,85},{5869,46},{5855,68},{5851,41},{5847,186},{5847,21},{5844,7},{5839,172},{5825,57},{5817,248},{5812,88},{5765,195},{5759,100},{5756,58},{5744,15},{5740,140},{5717,55},{5714,164},{5708,18},{5707,65},{5686,53},{5677,225},{5675,156},{5657,133},{5657,131},{5654,92},{5649,201},{5649,115},{5643,173},{5642,207},{5631,28},{5631,25},{5629,204},{5626,34},{5593,244},{5592,31},{5578,47},{5576,242},{5569,43},{5567,106},{5565,240},{5546,137},{5540,169},{5536,237},{5534,178},{5527,26},{5526,98},{5521,177},{5512,102},{5511,120},{5509,209},{5509,62},{5508,84},{5501,180},{5498,185},{5498,83},{5489,188},{5489,101},{5483,138},{5481,234},{5481,99},{5478,187},{5473,184},{5471,142},{5465,199},{5464,44},{5459,198},{5459,165},{5454,90},{5441,141},{5441,86},{5432,151},{5423,152},{5417,215},{5413,134},{5403,36},{5397,153},{5388,197},{5379,38},{5375,196},{5366,212},{5365,118},{5364,73},{5359,70},{5359,45},{5357,114},{5357,39},{5353,235},{5347,111},{5347,71},{5334,202},{5325,29},{5318,109},{5311,139},{5311,27},{5310,216},{5306,123},{5300,81},{5292,63},{5288,82},{5286,103},{5274,158},{5271,69},{5269,147},{5266,66},{5264,37},{5256,228},{5252,60},{5248,211},{5240,227},{5238,214},{5234,42},{5232,94},{5224,91},{5222,113},{5221,236},{5212,174},{5212,108},{5210,122},{5189,213},{5187,127},{5171,89},{5166,217},{5166,30},{5162,175},{5159,189},{5153,154},{5150,143},{5146,59},{5144,231},{5140,166},{5135,241},{5134,78},{5130,230},{5123,77},{5119,146},{5098,205},{5098,110},{5097,117},{5090,124},{5090,79},{5089,210},{5083,229},{5081,95},{5081,87},{5080,238},{5070,219},{5067,167},{5052,239},{5052,171},{5051,135},{5046,218},{5026,61},{5024,74},{5014,145},{5012,252},{5010,251},{5005,150},{5005,126},{4999,222},{4993,190},{4979,247},{4979,148},{4979,105},{4975,121},{4968,157},{4961,191},{4961,159},{4944,179},{4936,119},{4921,182},{4919,243},{4912,155},{4901,245},{4900,183},{4896,149},{4890,233},{4870,220},{4857,181},{4848,249},{4829,107},{4808,75},{4800,221},{4758,254},{4716,125},{4710,246},{4696,253},{4603,223}}},
{"cell_type",{{26368,0},{15157,1},{1641,190},{1232,32},{1139,12},{1032,114},{846,192},{787,82},{756,64},{711,189},{685,104},{611,16},{571,72},{516,185},{511,224},{509,80},{497,201},{475,160},{469,167},{407,223},{391,166},{381,4},{353,191},{334,128},{322,184},{288,102},{274,86},{236,83},{227,136},{203,98},{203,13},{191,115},{189,15},{187,14},{186,67},{182,52},{165,188},{164,66},{156,100},{133,117},{130,118},{126,23},{125,130},{125,108},{115,96},{107,134},{106,135},{104,68},{103,124},{102,119},{97,116},{89,2},{86,17},{85,221},{85,112},{77,125},{75,255},{75,127},{74,126},{60,8},{59,113},{57,122},{57,120},{53,176},{50,144},{49,84},{49,65},{48,59},{45,194},{44,20},{43,142},{42,5},{41,123},{39,101},{37,121},{35,237},{35,97},{35,49},{32,3},{31,161},{31,88},{30,212},{30,208},{30,74},{29,155},{27,70},{26,219},{25,178},{25,129},{25,48},{23,204},{23,6},{22,248},{22,81},{21,200},{20,175},{20,110},{20,18},{18,207},{18,109},{17,202},{17,62},{17,37},{16,209},{16,187},{16,183},{15,247},{15,173},{15,105},{15,71},{15,19},{14,249},{14,241},{14,195},{14,180},{14,163},{14,69},{13,242},{13,39},{12,169},{12,168},{12,151},{12,143},{12,73},{12,33},{11,228},{11,186},{11,162},{11,99},{10,196},{10,179},{10,95},{10,50},{9,41},{9,22},{8,227},{8,217},{8,147},{8,51},{7,240},{7,222},{7,182},{7,164},{7,106},{7,87},{7,24},{7,7},{6,216},{6,165},{6,140},{6,60},{6,56},{6,10},{5,244},{5,198},{5,181},{5,139},{5,46},{5,44},{4,250},{4,234},{4,213},{4,206},{4,205},{4,203},{4,152},{4,138},{4,111},{4,103},{4,85},{4,76},{4,9},{3,253},{3,220},{3,211},{3,210},{3,172},{3,158},{3,157},{3,107},{3,94},{3,89},{3,79},{3,75},{3,53},{3,35},{2,254},{2,252},{2,215},{2,174},{2,159},{2,154},{2,146},{2,141},{2,90},{2,54},{2,47},{1,243},{1,239},{1,235},{1,233},{1,226},{1,177},{1,170},{1,156},{1,149},{1,133},{1,132},{1,91},{1,78},{1,77},{1,63},{1,57},{1,28},{1,21},{0,251},{0,246},{0,245},{0,238},{0,236},{0,232},{0,231},{0,230},{0,229},{0,225},{0,218},{0,214},{0,199},{0,197},{0,193},{0,171},{0,153},{0,150},{0,148},{0,145},{0,137},{0,131},{0,93},{0,92},{0,61},{0,58},{0,55},{0,45},{0,43},{0,42},{0,40},{0,38},{0,36},{0,34},{0,31},{0,30},{0,29},{0,27},{0,26},{0,25},{0,11}}},
{"d1",{{28830,34},{14507,40},{7208,2},{4894,0},{4687,1},{1514,33},{1297,3},{570,35},{97,4},{84,8},{25,10},{24,36},{6,9}}},
{"d2",{{14521,72},{9099,15},{7528,17},{7104,13},{4147,11},{3057,1},{1378,9},{1031,105},{971,130},{909,111},{845,113},{820,19},{777,181},{772,7},{657,177},{558,81},{391,171},{385,67},{348,158},{284,75},{278,89},{277,163},{248,21},{214,161},{212,149},{210,151},{210,23},{210,3},{183,104},{156,175},{154,33},{153,201},{143,153},{143,115},{139,66},{134,152},{134,69},{131,225},{123,135},{120,109},{117,157},{111,156},{111,20},{110,97},{110,73},{109,150},{101,10},{94,91},{91,87},{90,25},{87,147},{86,12},{83,99},{76,16},{74,155},{73,162},{72,112},{66,154},{65,107},{64,80},{60,102},{59,179},{59,169},{55,98},{53,2},{49,121},{49,18},{48,47},{45,5},{43,8},{42,117},{42,65},{41,100},{41,27},{40,160},{38,197},{38,183},{37,229},{36,148},{35,88},{35,68},{34,217},{33,131},{33,37},{32,170},{30,138},{30,14},{29,176},{29,145},{29,143},{29,103},{29,74},{28,173},{28,137},{28,95},{28,94},{27,219},{27,185},{26,178},{25,247},{25,203},{25,172},{24,254},{24,244},{23,222},{23,32},{22,159},{22,55},{22,0},{21,233},{21,174},{21,79},{21,77},{21,26},{21,24},{20,180},{20,110},{20,85},{19,192},{19,167},{19,141},{19,76},{18,235},{17,191},{17,122},{17,71},{17,31},{16,255},{16,83},{15,246},{15,168},{15,132},{15,119},{14,35},{13,251},{13,242},{13,195},{13,165},{13,30},{13,4},{12,230},{12,215},{12,133},{12,114},{12,70},{11,241},{11,227},{11,127},{11,108},{11,22},{10,128},{10,124},{10,123},{10,101},{10,51},{10,34},{10,29},{9,231},{9,182},{9,126},{9,106},{9,38},{8,249},{8,248},{8,220},{8,142},{8,118},{8,64},{8,57},{8,53},{8,28},{7,236},{7,234},{7,208},{7,205},{7,193},{7,129},{7,120},{6,213},{6,202},{6,92},{6,78},{5,252},{5,243},{5,223},{5,212},{5,204},{5,200},{5,166},{5,139},{5,58},{5,56},{5,40},{4,245},{4,221},{4,216},{4,214},{4,209},{4,199},{4,194},{4,189},{4,86},{4,50},{4,46},{4,44},{4,6},{3,226},{3,210},{3,116},{3,96},{3,82},{3,61},{3,60},{3,48},{3,42},{3,39},{2,253},{2,238},{2,224},{2,196},{2,190},{2,187},{2,146},{2,144},{2,140},{2,136},{2,90},{2,84},{2,49},{2,45},{2,36},{1,250},{1,240},{1,232},{1,228},{1,218},{1,211},{1,206},{1,198},{1,184},{1,164},{1,134},{1,93},{1,62},{1,59},{1,54},{1,52},{0,239},{0,237},{0,207},{0,188},{0,186},{0,125},{0,63},{0,43},{0,41}}},
{"ref_diff",{{22819,0},{20114,1},{11193,2},{1646,3},{692,4},{575,5},{503,6},{446,7},{360,8},{318,9},{307,10},{270,11},{239,12},{208,13},{196,14},{187,15},{158,16},{117,20},{117,17},{114,19},{106,21},{103,18},{99,23},{97,22},{83,25},{81,31},{80,26},{78,24},{74,27},{71,28},{67,30},{65,29},{59,32},{58,35},{52,36},{46,37},{43,34},{42,33},{38,38},{36,45},{36,39},{34,46},{33,48},{33,44},{32,49},{32,40},{31,42},{29,66},{29,58},{29,41},{27,54},{26,519},{26,163},{26,57},{25,154},{25,129},{25,47},{24,603},{24,150},{24,124},{24,67},{24,65},{24,55},{24,43},{23,158},{23,115},{23,92},{23,68},{22,909},{22,241},{22,221},{22,156},{22,155},{22,144},{22,127},{22,125},{22,76},{22,53},{21,656},{21,210},{21,178},{21,170},{21,147},{21,133},{21,108},{21,101},{21,71},{21,70},{21,63},{21,61},{21,59},{20,904},{20,874},{20,854},{20,843},{20,623},{20,268},{20,258},{20,226},{20,222},{20,196},{20,177},{20,157},{20,142},{20,140},{20,120},{20,119},{20,116},{20,114},{20,113},{20,112},{20,90},{20,52},{19,841},{19,649},{19,613},{19,564},{19,229},{19,193},{19,167},{19,162},{19,152},{19,137},{19,131},{19,109},{19,94},{19,91},{19,73},{19,72},{19,69},{19,62},{19,60},{18,1089},{18,1053},{18,1044},{18,956},{18,890},{18,764},{18,664},{18,638},{18,626},{18,625},{18,604},{18,584},{18,507},{18,440},{18,356},{18,341},{18,291},{18,262},{18,238},{18,227},{18,204},{18,186},{18,166},{18,160},{18,159},{18,136},{18,135},{18,93},{18,88},{18,85},{18,80},{18,74},{18,51},{18,50},{17,963},{17,950},{17,940},{17,905},{17,851},{17,849},{17,660},{17,640},{17,621},{17,615},{17,552},{17,394},{17,370},{17,337},{17,244},{17,225},{17,208},{17,201},{17,194},{17,188},{17,184},{17,179},{17,164},{17,148},{17,132},{17,107},{17,102},{17,98},{17,97},{17,89},{17,79},{17,78},{17,75},{17,56},{16,1176},{16,1086},{16,1042},{16,977},{16,967},{16,958},{16,943},{16,918},{16,915},{16,865},{16,862},{16,855},{16,674},{16,631},{16,628},{16,624},{16,619},{16,605},{16,594},{16,581},{16,541},{16,522},{16,521},{16,492},{16,489},{16,475},{16,458},{16,425},{16,421},{16,387},{16,360},{16,342},{16,326},{16,272},{16,248},{16,213},{16,199},{16,195},{16,165},{16,146},{16,128},{16,126},{16,123},{16,122},{16,104},{16,83},{15,1360},{15,1311},{15,1303},{15,1281},{15,1181},{15,1179},{15,1138},{15,1114},{15,1094},{15,1092}}},
};




struct HuffmanEncoder {
  HuffmanEncoder() {}
  HuffmanEncoder(const distribution_data& data, const std::string name) {
    DBG(log_level::ENCODER_BUILD, data);
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
    MSG(log_level::ENCODER_BUILD, "Huffman encoder data for ", name, ": ", debug::dbgout(uncompressed, compressed, compression_ratio));
  }
  template<class Writer>
  void write(BitWriter<Writer>& bwriter, int value) const {
    auto [code, len] = code_len.at(value);
    MSG(log_level::NUMBER, "Huffman write ", code, ' ', len, ' ', value);
    bwriter.write_bits(code, len);
  }
  int read(BitReader& breader) const {
    uint64_t code = 0;
    for (int b = 0; b < 64; ++b) {
      code |= static_cast<uint64_t>(breader.read_bit()) << b;
      // code = (code << 1) + breader.read_bit();
      auto it = code_index.find({code, b + 1});
      if (it != code_index.end()) {
        MSG(log_level::NUMBER, "Huffman read ", code, ' ', b + 1, ' ', it->second);
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
    DCHECK(data.size() > 1);
    std::vector<std::pair<long long, std::vector<int>>> by_cnt;
    for (auto [cnt, value] : data) {
      by_cnt.push_back({cnt, {value}});
    }
    // TODO: improve performance for bigger data
    while (by_cnt.size() > 1) {
      std::sort(by_cnt.begin(), by_cnt.end(), [](auto lhs, auto rhs) {
        if (lhs.first != rhs.first) return lhs.first > rhs.first;
        return lhs.second.size() > rhs.second.size();
      });
      auto [c1, v1] = by_cnt.back();
      by_cnt.pop_back();
      auto [c2, v2] = by_cnt.back();
      by_cnt.pop_back();
      for (auto x : v1) add_bit(x, 0);
      for (auto x : v2) add_bit(x, 1);
      v1.insert(v1.end(), v2.begin(), v2.end());
      c1 += c2;
      by_cnt.push_back({c1, v1});
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
    DBG(log_level::ENCODER_BUILD, name, special, min_value, extra_bits);
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

static const HuffmanEncoder d1(huffman_data.at("d1"), "d1");
static const HuffmanEncoder d2(huffman_data.at("d2"), "d2");
static const HuffmanEncoder cell_data(huffman_data.at("cell_data"), "cell_data");
static const HuffmanEncoder cell_type(huffman_data.at("cell_type"), "cell_type");
HuffmanEncoderWithDefault ref_diff;

void init_ref_diff(int cell_count) {
  int l = 0;
  int r = std::max(cell_count, 1);
  ref_diff = HuffmanEncoderWithDefault(huffman_data.at("ref_diff"), -1, {l, r}, "ref_diff");
}

} // namespace huffman

namespace vm {


struct CustomCellSerializationInfo : public CellSerializationInfo {
  td::Result<int> custom_get_bits(td::Slice cell_data) const {
      if (data_with_bits) {
        // for (int i = 0; i <= data_offset + data_len - 1; ++i) {
        //   std::cerr << uint16_t(uint8_t(cell[i])) << ' ';
        // }
        // std::cerr << '\n';
        // std::cerr << "Data offsets " << data_offset << ' ' << data_len << '\n';
        DCHECK(data_len != 0);
        int last = cell_data[data_len - 1];
        if (!(last & 0x7f)) {
          return td::Status::Error("overlong encoding");
        }
        return td::narrow_cast<int>((data_len - 1) * 8 + 7 - td::count_trailing_zeroes_non_zero32(last));
      } else {
        return td::narrow_cast<int>(data_len * 8);
      }
  }
  td::Status custom_init(td::Slice data, int ref_bit_size) {
    if (data.size() < 2) {
      return td::Status::Error(PSLICE() << "Not enough bytes " << td::tag("got", data.size())
                                        << td::tag("expected", "at least 2"));
    }
    TRY_STATUS(custom_init(data.ubegin()[0], data.ubegin()[1], ref_bit_size));
    if (data.size() < end_offset) {
      return td::Status::Error(PSLICE() << "Not enough bytes " << td::tag("got", data.size())
                                        << td::tag("expected", end_offset));
    }
    return td::Status::OK();
  }

  td::Status custom_init(td::uint8 d1, td::uint8 d2, int ref_bit_size) {
    refs_cnt = d1 & 7;
    level_mask = Cell::LevelMask(d1 >> 5);
    special = (d1 & 8) != 0;
    with_hashes = (d1 & 16) != 0;

    if (refs_cnt > 4) {
      if (refs_cnt != 7 || !with_hashes) {
        return td::Status::Error("Invalid first byte");
      }
      refs_cnt = 0;
      // ...
      // do not deserialize absent cells!
      return td::Status::Error("TODO: absent cells");
    }

    data_len = (d2 >> 1) + (d2 & 1);
    data_with_bits = (d2 & 1) != 0;

    return td::Status::OK();
  }
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
};

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
    // unsigned long long read_int(const unsigned char* ptr, unsigned bytes);
    // unsigned long long read_ref(const unsigned char* ptr) {
    //   return read_int(ptr, ref_bit_size);
    // }
    // unsigned long long read_offset(const unsigned char* ptr) {
    //   return read_int(ptr, offset_bit_size);
    // }
    // void write_int(unsigned char* ptr, unsigned long long value, int bytes);
    // void write_ref(unsigned char* ptr, unsigned long long value) {
    //   write_int(ptr, value, ref_bit_size);
    // }
    // void write_offset(unsigned char* ptr, unsigned long long value) {
    //   write_int(ptr, value, offset_bit_size);
    // }
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
  void clear();
  int set_roots(const std::vector<td::Ref<vm::Cell>>& new_roots);
  int set_root(td::Ref<vm::Cell> new_root);
  int add_roots(const std::vector<td::Ref<vm::Cell>>& add_roots);
  int add_root(td::Ref<vm::Cell> add_root);
  td::Status import_cells() TD_WARN_UNUSED_RESULT;
  CustomBagOfCells() = default;
  std::size_t estimate_serialized_size();
  td::Status serialize(int mode = 0);
  td::string serialize_to_string(int mode = 0);
  td::Result<td::BufferSlice> serialize_to_slice();
  td::Result<std::size_t> serialize_to(unsigned char* buffer, std::size_t buff_size);
  td::Status serialize_to_file(td::FileFd& fd, int mode = 0);
  template <typename WriterT>
  td::Result<std::size_t> serialize_to_impl(WriterT& writer);
  std::string extract_string() const;

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

  static int precompute_cell_serialization_size(const unsigned char* cell, std::size_t len, int ref_size,
                                                int* refs_num_ptr = nullptr);

  int rv_idx;
  td::Result<int> import_cell(td::Ref<vm::Cell> cell, int depth);
  void cells_clear() {
    cell_count = 0;
    int_refs = 0;
    data_bytes = 0;
    cells.clear();
    cell_list_.clear();
  }
  td::uint64 compute_sizes(int& r_size);
  void reorder_cells();
  int revisit(int cell_idx, int force = 0);
  unsigned long long get_idx_entry_raw(int index);
  unsigned long long get_idx_entry(int index);
  bool get_cache_entry(int index);
  td::Result<td::Slice> get_cell_slice(int index, td::Slice data);
  td::Result<td::Ref<vm::DataCell>> deserialize_cell(int idx, td::Span<td::Ref<DataCell>> cells, BitReader& breader, CustomCellSerializationInfo& cell_info);
};

// unsigned long long CustomBagOfCells::Info::read_int(const unsigned char* ptr, unsigned bytes) {
//   unsigned long long res = 0;
//   while (bytes > 0) {
//     res = (res << 8) + *ptr++;
//     --bytes;
//   }
//   return res;
// }
// void CustomBagOfCells::Info::write_int(unsigned char* ptr, unsigned long long value, int bytes) {
//   ptr += bytes;
//   while (bytes) {
//     *--ptr = value & 0xff;
//     value >>= 8;
//     --bytes;
//   }
//   DCHECK(!bytes);
// }

long long CustomBagOfCells::Info::parse_serialized_header(BitReader& breader) {
  invalidate();
  ref_bit_size = 0;
  root_count = cell_count = -1;
  index_offset = data_offset = total_size = 0;
  // if (sz < 1) {
  //   return -10;
  // }
  // ref_bit_size = 2;
  ref_bit_size = breader.read_bits(5);
  if (ref_bit_size > 4 * 8 || ref_bit_size < 1) {
    return 0;
  }
  // if (sz < 6) {
  //   return -7 - 3 * ref_bit_size;
  // }
  // offset_bit_size = ptr[1];

  DBG(log_level::COMPRESSION_META, ref_bit_size);
  auto read_ref = [&]() -> uint64_t {
    return breader.read_bits(ref_bit_size);
  };
  cell_count = (int)read_ref();
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

//serialized_boc#672fb0ac has_idx:(## 1) has_crc32c:(## 1)
//  has_cache_bits:(## 1) flags:(## 2) { flags = 0 }
//  size:(## 3) { size <= 4 }
//  off_bytes:(## 8) { off_bytes <= 8 }
//  cells:(##(size * 8))
//  roots:(##(size * 8))
//  absent:(##(size * 8)) { roots + absent <= cells }
//  tot_cells_size:(##(off_bytes * 8))
//  index:(cells * ##(off_bytes * 8))
//  cell_data:(tot_cells_size * [ uint8 ])
//  = BagOfCells;
// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
template <typename WriterT>
td::Result<std::size_t> CustomBagOfCells::serialize_to_impl(WriterT& writer) {
  BitWriter bwriter(writer);
  MSG(log_level::COMPRESSION_META, "Running custom serialize impl");
  auto store_ref = [&](unsigned long long value) { bwriter.write_bits(value, info.ref_bit_size); };

  td::uint8 byte{0};
  // 3, 4 - flags
  if (info.ref_bit_size < 1 || info.ref_bit_size > 4 * 8) {
    return 0;
  }
  bwriter.write_bits(info.ref_bit_size, 5);


  store_ref(cell_count);
  huffman::init_ref_diff(cell_count);
  // DCHECK(writer.position() == info.index_offset);
  DCHECK((unsigned)cell_count == cell_list_.size());
  // DCHECK(writer.position() == info.data_offset);
  // bwriter.flush_byte();

  std::vector<std::vector<uint8_t>> serialized_cells(cell_count, std::vector<uint8_t>(256));

  for (int i = cell_count - 1; i >= 0; --i) {
    int idx = cell_count - 1 - i;
    const auto& dc_info = cell_list_[idx];
    const Ref<DataCell>& dc = dc_info.dc_ref;
    MSG(log_level::CELL_META, "Serializing cell with idx ", i, " with refnum ", int(dc_info.ref_num));
    int s = dc->serialize(serialized_cells[i].data(), 256);
    MSG(log_level::CELL_META, "Cell serialized size = ", s);
    serialized_cells[i].resize(s);
  }

  for (int i = cell_count - 1; i >= 0; --i) {
    int idx = cell_count - 1 - i;
    MSG(log_level::CELL_META, "Saving cell with idx ", i, " with refnum ", int(cell_list_[idx].ref_num));
    auto start_position = bwriter.position();
    const auto& dc_info = cell_list_[idx];
    const Ref<DataCell>& dc = dc_info.dc_ref;
    int s = serialized_cells[i].size();
    auto* buf = serialized_cells[i].data();
    MSG(log_level::CELL_META, "Cell d1, d2 = ", uint16_t(buf[0]), ' ', uint16_t(buf[1]));
    // writer.store_bytes(buf, s);
    auto store_cell_data = [&]() {
      bwriter.flush_byte();
      for (int i = 3; i < s; ++i) {
        MSG(log_level::LOG_LEVEL::BYTE, "Saving byte ", uint16_t(buf[i]));
        add_char("cell_data", buf[i]);
        bwriter.write_bits(buf[i], 8);
        // huffman::cell_data.write(bwriter, buf[i]);
      }
      // bwriter.flush_byte();
    };
    auto store_cell_refs = [&]() {
      DCHECK(dc->size_refs() == dc_info.ref_num);
      for (unsigned j = 0; j < dc_info.ref_num; ++j) {
        int k = cell_count - 1 - dc_info.ref_idx[j];
        MSG(log_level::CELL_META, "Link from ", i, " to ", k);
        DCHECK(k > i && k < cell_count);
        int ref_diff = k - i - 1;
        add_int("ref_diff", ref_diff);
        // huffman::ref_diff.write(bwriter, ref_diff);
        store_ref(ref_diff);
      }
    };
    auto store_prunned_branch = [&]() {
      int l = s - 4;
      DCHECK(l % 34 == 0);
      l /= 34;
      for (int x = 0; x < l; ++x) {
        int id = 4 + 32 * l + 2 * x;
        int val = (uint16_t(buf[id]) << 8) | uint16_t(buf[id + 1]);
        add_int("byte_depth", val);
      }
      // l /= 34;
      // DCHECK(buf[3] <= 3);
      store_cell_data();
    };
    uint16_t d1 = buf[0];
    bool is_special = d1 & 8;
    uint16_t d2 = buf[1];
    huffman::d1.write(bwriter, d1);
    huffman::d2.write(bwriter, d2);
    uint16_t cell_type = buf[2];
    huffman::cell_type.write(bwriter, cell_type);
    add_char("cell_type", cell_type);
    if (is_special && cell_type == 1) {
      store_prunned_branch();
    } else {
      store_cell_data();
    }
    store_cell_refs();
    auto end_position = bwriter.position();
    MSG(log_level::CELL_META, "Cell position ", start_position, ' ', end_position);
  }
  bwriter.flush_byte(); // It's important to write the last byte, otherwise it will stay in the buffer
  writer.chk();
  // DCHECK(writer.position() - keep_position == info.data_size);
  // DCHECK(writer.empty());
  DBG(log_level::COMPRESSION_META, writer.position());
  return writer.position();
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
td::uint64 CustomBagOfCells::compute_sizes(int& r_size) {
  int rs = 0;
  if (!root_count || !data_bytes) {
    r_size = 0;
    return 0;
  }
  while (cell_count >= (1LL << rs)) {
    rs++;
  }
  td::uint64 data_bytes_adj = cell_count * 3 + data_bytes + (unsigned long long)int_refs * ((rs + 7) / 8);
  if (rs > 4 * 8) {
    r_size = 0;
    return 0;
  }
  r_size = rs;
  return data_bytes_adj;
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
std::size_t CustomBagOfCells::estimate_serialized_size() {
  auto data_bytes_adj = compute_sizes(info.ref_bit_size);
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
  std::size_t size_est = estimate_serialized_size();
  if (!size_est || size_est > buff_size) {
    return 0;
  }
  boc_writers::BufferWriter writer{buffer, buffer + size_est};
  return serialize_to_impl(writer);
}

td::Result<td::BufferSlice> CustomBagOfCells::serialize_to_slice() {
  std::size_t size_est = estimate_serialized_size();
  if (!size_est) {
    return td::Status::Error("no cells to serialize to this bag of cells");
  }
  int buff_size = 2 << 20;
  td::BufferSlice res(buff_size);
  TRY_RESULT(size, serialize_to(const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(res.data())),
                                res.size()));
  MSG(log_level::COMPRESSION_META, "Expected size = ", size_est, ", Real size = ", size);
  if (size <= res.size()) {
    res.truncate(size);
    return std::move(res);
  } else {
    return td::Status::Error("error while serializing a bag of cells: actual serialized size differs from estimated");
  }
}

td::Result<td::Ref<vm::DataCell>> CustomBagOfCells::deserialize_cell(int idx, td::Span<td::Ref<DataCell>> cells_span,
                                                                     BitReader& breader, CustomCellSerializationInfo& cell_info) {

  CellBuilder cb;
  td::BufferSlice cell_slice(cell_info.data_len);

  auto load_cell_data = [&]() {
    breader.flush_byte();
    for (int i = 1; i < cell_info.data_len; ++i) {
      uint8_t byte = breader.read_bits(8);
      // uint8_t byte = huffman::cell_data.read(breader);
      cell_slice.data()[i] = byte;
    }
    TRY_RESULT(bits, cell_info.custom_get_bits(cell_slice));
    MSG(log_level::CELL_META, "Cell bits size = ", bits);
    cb.store_bits(cell_slice.data(), bits);

    // breader.flush_byte();
    return td::Status::OK();
  };

  std::array<td::Ref<Cell>, 4> refs_buf;
  auto refs = td::MutableSpan<td::Ref<Cell>>(refs_buf).substr(0, cell_info.refs_cnt);

  auto load_cell_refs = [&]() {
    DBG(log_level::CELL_META, cell_info.refs_cnt);

    for (int k = 0; k < cell_info.refs_cnt; k++) {
      // int ref_diff = huffman::ref_diff.read(breader);
      int ref_diff = breader.read_bits(info.ref_bit_size);
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
      refs[k] = cells_span[cell_count - ref_idx - 1];
    }
    return td::Status::OK();
  };

  cell_slice.data()[0] = huffman::cell_type.read(breader);
  auto cell_data_res = load_cell_data();
  if (cell_data_res.is_error()) {
    return cell_data_res;
  }
  auto cell_refs_res = load_cell_refs();
  if (cell_refs_res.is_error()) {
    return cell_refs_res;
  }

  return cell_info.custom_create_data_cell(cb, refs);
}

td::Result<long long> CustomBagOfCells::deserialize(const td::Slice& data, int max_roots) {
  MSG(log_level::COMPRESSION_META, "Running custom deserialize impl");
  // for (int i = 0; i < data.size(); ++i) {
  //   std::cerr << uint16_t(uint8_t(data[i])) << ' ';
  // }
  // std::cerr << '\n';
  get_og().clear();
  BitReader breader(data, 0);
  long long start_offset = info.parse_serialized_header(breader);
  if (start_offset == 0) {
    return td::Status::Error(PSLICE() << "cannot deserialize bag-of-cells: invalid header, error " << start_offset);
  }
  if (start_offset < 0) {
    return start_offset;
  }

  //LOG(INFO) << "estimated size " << size_est << ", true size " << data.size();
  if (info.root_count > max_roots) {
    return td::Status::Error("Bag-of-cells has more root cells than expected");
  }

  auto read_ref = [&]() -> uint64_t {
    return breader.read_bits(info.ref_bit_size);
  };

  cell_count = info.cell_count;
  roots.clear();
  roots.resize(info.root_count);
  roots[0].idx = info.cell_count - 1;
  std::vector<Ref<DataCell>> cell_list;
  cell_list.reserve(cell_count);
  for (int i = 0; i < cell_count; i++) {
    CustomCellSerializationInfo cell_info;
    auto d1 = huffman::d1.read(breader);
    auto d2 = huffman::d2.read(breader);
    add_char("d1", d1);
    add_char("d2", d2);
    DBG(log_level::CELL_META, d1, d2);
    auto status = cell_info.custom_init(d1, d2, info.ref_bit_size);
    if (status.is_error()) {
      return td::Status::Error(PSLICE()
                                << "invalid bag-of-cells failed to deserialize cell #" << i << " " << status.error());
    }
    // reconstruct cell with index cell_count - 1 - i
    int idx = cell_count - 1 - i;
    MSG(log_level::CELL_META, "Loading cell with idx ", idx);
    auto r_cell = deserialize_cell(idx, cell_list, breader, cell_info);
    if (r_cell.is_error()) {
      return td::Status::Error(PSLICE() << "invalid bag-of-cells failed to deserialize cell #" << idx << " "
                                        << r_cell.error());
    }
    cell_list.push_back(r_cell.move_as_ok());
    DCHECK(cell_list.back().not_null());
  }
  auto end_offset = breader.flush_and_get_ptr();
  index_ptr = nullptr;
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

enum class FinalCompression {
  ORIGINAL,
  LZ4,
  DEFLATE
};

static constexpr enum FinalCompression final_compression = FinalCompression::DEFLATE;

td::BufferSlice apply_final_compression(td::Slice data) {
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

td::BufferSlice invert_final_compression(td::Slice data) {
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

td::BufferSlice applyBWT(td::Slice data) {
  auto [bwt_result, special_symbol_pos] = BWT::bwt(BWT::to_byte_buffer(data));
  BWT::byte_buffer special_sumbol_bytes = {
    special_symbol_pos & 255,
    special_symbol_pos >> 8 & 255,
    special_symbol_pos >> 16 & 255
  };
  bwt_result.insert(bwt_result.begin(), special_sumbol_bytes.begin(), special_sumbol_bytes.end());
  return std::move(BWT::from_byte_buffer(bwt_result));
}

td::BufferSlice inverseBWT(td::Slice data) {
  auto ptr = data.ubegin();
  int special_symbol_pos = (int(ptr[0]) << 0) + (int(ptr[1]) << 8) + (int(ptr[2]) << 16);
  data.remove_prefix(3);
  auto inverse_bwt = BWT::inverse_bwt(BWT::to_byte_buffer(data), special_symbol_pos);
  return std::move(BWT::from_byte_buffer(inverse_bwt));
}

bool use_bwt = false;

td::BufferSlice compress(td::Slice data) {
  td::Ref<vm::Cell> root = vm::std_boc_deserialize(data).move_as_ok();
  td::BufferSlice serialized = vm::custom_boc_serialize(root).move_as_ok();
  auto with_bwt = use_bwt ? td::BufferSlice(applyBWT(std::move(serialized))) : std::move(serialized);
  return apply_final_compression(std::move(with_bwt));
}

td::BufferSlice decompress(td::Slice data) {
  auto decompressed = invert_final_compression(data);
  auto without_bwt = use_bwt ? inverseBWT(std::move(decompressed)) : std::move(decompressed);
  vm::Ref<vm::Cell> root = vm::custom_boc_deserialize(without_bwt).move_as_ok();
  return vm::std_boc_serialize(root, 31).move_as_ok();
}

int main() {
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
      for (auto [value, count] : st_data) {
        std::cout << name << ' ' << value << ' ' << count << '\n';
      }
    }
  #endif
}

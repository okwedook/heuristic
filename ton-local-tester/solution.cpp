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

#ifndef ONLINE_JUDGE
    void flush() { std::cerr << std::flush; }
    void flushln() { std::cerr << std::endl; }
    template<class T> void print(const T &x) { std::cerr << x; }
    template<class T, class ...U> void print(const T &x, const U&... u) { print(x); print(u...); }
    template<class ...T> void println(const T&... u) { print(u..., '\n'); }
    #define dbg(...) print("[", #__VA_ARGS__, "] = ", debug::dbgout(__VA_ARGS__)), flushln()
    #define msg(...) print("[", __VA_ARGS__, "]"), flushln()
#else
    #define dbg(...) 0
    #define msg(...) 0
#endif

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

    // std::vector<byte_buffer> rotations(modified_n);
    
    // // Create all rotations of the modified input string
    // for (size_t i = 0; i < modified_n; ++i) {
    //     byte_buffer v(modified_input.begin() + i, modified_input.end());
    //     byte_buffer c(modified_input.begin(), modified_input.begin() + i);
    //     v.insert(v.end(), c.begin(), c.end());
    //     rotations[i] = v;
    // }

    // // Sort the rotations
    // std::sort(rotations.begin(), rotations.end());
    // dbg(rotations);

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
    // std::cerr << "Writing bits " << value << ' ' << bit_size << '\n';
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
      // std::cerr << "Flush byte\n";
      w.store_uint(bit_value, 1); // stores exactly byte
      bits = 0;
      bit_value = 0;
    }
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
    // std::cerr << "Reading bits " << ans << ' ' << bits << '\n';
    return ans;
  }
  int flush_and_get_ptr() {
    flush_byte();
    return ptr;
  }
  void flush_byte() {
    if (bit_index != 0) {
      // std::cerr << "Flush byte\n";
      ++ptr;
      bit_index = 0;
    }
  }
private:
  const td::Slice& data;
  int ptr;
  int bit_index;
};


namespace huffman {

using distribution_data = std::vector<std::pair<long long, int>>;

static const std::map<std::string, distribution_data> huffman_data = {
{"d1",{{28830,34},{14507,40},{7208,2},{4894,0},{4687,1},{1514,33},{1297,3},{570,35},{97,4},{84,8},{25,10},{24,36},{6,9}}},
{"d2",{{14521,72},{9099,15},{7528,17},{7104,13},{4147,11},{3057,1},{1378,9},{1031,105},{971,130},{909,111},{845,113},{820,19},{777,181},{772,7},{657,177},{558,81},{391,171},{385,67},{348,158},{284,75},{278,89},{277,163},{248,21},{214,161},{212,149},{210,151},{210,23},{210,3},{183,104},{156,175},{154,33},{153,201},{143,153},{143,115},{139,66},{134,152},{134,69},{131,225},{123,135},{120,109},{117,157},{111,156},{111,20},{110,97},{110,73},{109,150},{101,10},{94,91},{91,87},{90,25},{87,147},{86,12},{83,99},{76,16},{74,155},{73,162},{72,112},{66,154},{65,107},{64,80},{60,102},{59,179},{59,169},{55,98},{53,2},{49,121},{49,18},{48,47},{45,5},{43,8},{42,117},{42,65},{41,100},{41,27},{40,160},{38,197},{38,183},{37,229},{36,148},{35,88},{35,68},{34,217},{33,131},{33,37},{32,170},{30,138},{30,14},{29,176},{29,145},{29,143},{29,103},{29,74},{28,173},{28,137},{28,95},{28,94},{27,219},{27,185},{26,178},{25,247},{25,203},{25,172},{24,254},{24,244},{23,222},{23,32},{22,159},{22,55},{22,0},{21,233},{21,174},{21,79},{21,77},{21,26},{21,24},{20,180},{20,110},{20,85},{19,192},{19,167},{19,141},{19,76},{18,235},{17,191},{17,122},{17,71},{17,31},{16,255},{16,83},{15,246},{15,168},{15,132},{15,119},{14,35},{13,251},{13,242},{13,195},{13,165},{13,30},{13,4},{12,230},{12,215},{12,133},{12,114},{12,70},{11,241},{11,227},{11,127},{11,108},{11,22},{10,128},{10,124},{10,123},{10,101},{10,51},{10,34},{10,29},{9,231},{9,182},{9,126},{9,106},{9,38},{8,249},{8,248},{8,220},{8,142},{8,118},{8,64},{8,57},{8,53},{8,28},{7,236},{7,234},{7,208},{7,205},{7,193},{7,129},{7,120},{6,213},{6,202},{6,92},{6,78},{5,252},{5,243},{5,223},{5,212},{5,204},{5,200},{5,166},{5,139},{5,58},{5,56},{5,40},{4,245},{4,221},{4,216},{4,214},{4,209},{4,199},{4,194},{4,189},{4,86},{4,50},{4,46},{4,44},{4,6},{3,226},{3,210},{3,116},{3,96},{3,82},{3,61},{3,60},{3,48},{3,42},{3,39},{2,253},{2,238},{2,224},{2,196},{2,190},{2,187},{2,146},{2,144},{2,140},{2,136},{2,90},{2,84},{2,49},{2,45},{2,36},{1,250},{1,240},{1,232},{1,228},{1,218},{1,211},{1,206},{1,198},{1,184},{1,164},{1,134},{1,93},{1,62},{1,59},{1,54},{1,52}}},
};

struct HuffmanEncoder {
  HuffmanEncoder(const distribution_data& data) {
    set_data(data);
    #ifndef ONLINE_JUDGE
      eval_data(data);
    #endif
  }
  void eval_data(const distribution_data& data) {
    msg("Eval data");
    dbg(code_len);
    long long uncompressed = 0, compressed = 0;
    for (auto [count, value] : data) {
      auto [_, len] = code_len.at(value);
      uncompressed += 8 * count;
      compressed += len * count;
    }
    auto compression_ratio = uncompressed * 1.0 / compressed;
    msg("Huffman encoder data: ", debug::dbgout(uncompressed, compressed, compression_ratio));
  }
  // template<class Writer>
  // void write(BitWriter<Writer>& bwriter, int value) const {
  //   auto [code, len] = code_len.at(value);
  //   bwriter.write_bits(code, len);
  // }
  // int read(BitReader& breader) const {
  //   uint64_t 
  //   for (int b = 0; b < 64; ++b) {

  //   }
  // }
protected:
  void set_data(const distribution_data& data) {
    DCHECK(data.size() > 1);
    std::vector<std::pair<long long, std::vector<int>>> by_cnt;
    for (auto [cnt, value] : data) {
      by_cnt.push_back({cnt, {value}});
    }
    while (by_cnt.size() > 1) {
      std::sort(by_cnt.rbegin(), by_cnt.rend());
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
  std::map<int, std::pair<uint64_t, int>> code_len;
private:
  void add_bit(int value, int bit) {
    auto& [code, len] = code_len[value];
    code = (code << 1) + bit;
    ++len;
  }
};

static const HuffmanEncoder d1(huffman_data.at("d1"));
static const HuffmanEncoder d2(huffman_data.at("d2"));

} // namespace huffman

namespace vm {


struct CustomCellSerializationInfo : public CellSerializationInfo {

  td::Result<int> custom_get_bits(td::Slice cell) const {
      if (data_with_bits) {
        // for (int i = 0; i <= data_offset + data_len - 1; ++i) {
        //   std::cerr << uint16_t(uint8_t(cell[i])) << ' ';
        // }
        // std::cerr << '\n';
        // std::cerr << "Data offsets " << data_offset << ' ' << data_len << '\n';
        DCHECK(data_len != 0);
        int last = cell[data_offset + data_len - 1];
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

    hashes_offset = 2;
    auto n = level_mask.get_hashes_count();
    depth_offset = hashes_offset;
    data_offset = depth_offset;
    data_len = (d2 >> 1) + (d2 & 1);
    data_with_bits = (d2 & 1) != 0;
    refs_offset = data_offset + data_len;
    end_offset = refs_offset + (refs_cnt * ref_bit_size + 7) / 8;

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
  td::uint64 compute_sizes(int& r_size, int& o_size);
  void reorder_cells();
  int revisit(int cell_idx, int force = 0);
  unsigned long long get_idx_entry_raw(int index);
  unsigned long long get_idx_entry(int index);
  bool get_cache_entry(int index);
  td::Result<td::Slice> get_cell_slice(int index, td::Slice data);
  td::Result<td::Ref<vm::DataCell>> deserialize_cell(int idx, td::Slice cell_slice, td::Span<td::Ref<DataCell>> cells, BitReader& breader, CustomCellSerializationInfo& cell_info);
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
  // int sz = static_cast<int>(std::min(slice.size(), static_cast<std::size_t>(0xffff)));
  // magic = (unsigned)read_int(ptr, 4);
  magic = boc_generic;
  has_crc32c = false;
  has_index = false;
  has_cache_bits = false;
  ref_bit_size = 0;
  offset_bit_size = 0;
  root_count = cell_count = absent_count = -1;
  index_offset = data_offset = data_size = total_size = 0;
  if (magic != boc_generic && magic != boc_idx && magic != boc_idx_crc32c) {
    magic = 0;
    return 0;
  }
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
  offset_bit_size = breader.read_bits(5);

  dbg(ref_bit_size, offset_bit_size);
  if (offset_bit_size > 4 * 8 || offset_bit_size < 1) {
    return 0;
  }
  auto read_ref = [&]() -> uint64_t {
    return breader.read_bits(ref_bit_size);
  };
  auto read_offset = [&]() -> uint64_t {
    return breader.read_bits(offset_bit_size);
  };
  cell_count = (int)read_ref();
  if (cell_count <= 0) {
    cell_count = -1;
    return 0;
  }
  root_count = (int)read_ref();
  if (root_count <= 0) {
    root_count = -1;
    return 0;
  }
  has_roots = true;
  absent_count = (int)read_ref();
  if (absent_count < 0 || absent_count > cell_count) {
    return 0;
  }
  data_size = read_offset();
  if (data_size > ((unsigned long long)cell_count << 10)) {
    return 0;
  }
  if (data_size > (1ull << 40)) {
    return 0;  // bag of cells with more than 1TiB data is unlikely
  }
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
  msg("Running custom serialize impl");
  auto store_ref = [&](unsigned long long value) { bwriter.write_bits(value, info.ref_bit_size); };
  auto store_offset = [&](unsigned long long value) { bwriter.write_bits(value, info.offset_bit_size); };

  td::uint8 byte{0};
  // 3, 4 - flags
  if (info.ref_bit_size < 1 || info.ref_bit_size > 4 * 8) {
    return 0;
  }
  bwriter.write_bits(info.ref_bit_size, 5);

  bwriter.write_bits(info.offset_bit_size, 5);

  store_ref(cell_count);
  store_ref(root_count);
  store_ref(0);
  store_offset(info.data_size);
  for (const auto& root_info : roots) {
    int k = cell_count - 1 - root_info.idx;
    DCHECK(k >= 0 && k < cell_count);
    store_ref(k);
  }
  // DCHECK(writer.position() == info.index_offset);
  DCHECK((unsigned)cell_count == cell_list_.size());
  // DCHECK(writer.position() == info.data_offset);
  size_t keep_position = writer.position();
  bwriter.flush_byte();
  for (int i = cell_count - 1; i >= 0; --i) {
    int idx = cell_count - 1 - i;
    msg("Saving cell with idx ", i, " with refnum ", int(cell_list_[idx].ref_num));
    auto start_position = writer.position();
    const auto& dc_info = cell_list_[idx];
    const Ref<DataCell>& dc = dc_info.dc_ref;
    unsigned char buf[256];
    int s = dc->serialize(buf, 256);
    msg("Cell serialized size = ", s);
    msg("Cell d1, d2 = ", uint16_t(buf[0]), ' ', uint16_t(buf[1]));
    // writer.store_bytes(buf, s);
    for (int i = 0; i < s; ++i) {
      // std::cerr << "Saving byte " << uint16_t(buf[i]) << '\n';
      bwriter.write_bits(buf[i], 8);
    }
    DCHECK(dc->size_refs() == dc_info.ref_num);
    for (unsigned j = 0; j < dc_info.ref_num; ++j) {
      int k = cell_count - 1 - dc_info.ref_idx[j];
      msg("Link from ", i, " to ", k);
      DCHECK(k > i && k < cell_count);
      store_ref(k - i - 1);
    }
    bwriter.flush_byte();
    auto end_position = writer.position();
    msg("Cell position ", start_position, ' ', end_position);
  }
  writer.chk();
  // DCHECK(writer.position() - keep_position == info.data_size);
  // DCHECK(writer.empty());
  dbg(writer.position());
  return writer.position();
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
td::uint64 CustomBagOfCells::compute_sizes(int& r_size, int& o_size) {
  int rs = 0, os = 0;
  if (!root_count || !data_bytes) {
    r_size = o_size = 0;
    return 0;
  }
  while (cell_count >= (1LL << rs)) {
    rs++;
  }
  td::uint64 data_bytes_adj = data_bytes + (unsigned long long)int_refs * ((rs + 7) / 8);
  td::uint64 max_offset = data_bytes_adj;
  while (max_offset >= (1ULL << os)) {
    os++;
  }
  if (rs > 4 * 8 || os > 4 * 8) {
    r_size = o_size = 0;
    return 0;
  }
  r_size = rs;
  o_size = os;
  return data_bytes_adj;
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
std::size_t CustomBagOfCells::estimate_serialized_size() {
  auto data_bytes_adj = compute_sizes(info.ref_bit_size, info.offset_bit_size);
  dbg(info.ref_bit_size, info.offset_bit_size);
  if (!data_bytes_adj) {
    info.invalidate();
    return 0;
  }
  info.valid = true;
  info.has_crc32c = false;
  info.has_index = false;
  info.has_cache_bits = false;
  info.root_count = root_count;
  info.cell_count = cell_count;
  info.absent_count = dangle_count;
  info.roots_offset = 0 + 1 + 0 + 3 * ((info.ref_bit_size + 7) / 8) + (info.offset_bit_size + 7) / 8;
  info.index_offset = info.roots_offset + info.root_count * (info.ref_bit_size + 7) / 8;
  info.data_offset = info.index_offset;
  info.magic = Info::boc_generic;
  info.data_size = data_bytes_adj;
  info.total_size = info.data_offset + data_bytes_adj;
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
  td::BufferSlice res(size_est);
  TRY_RESULT(size, serialize_to(const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(res.data())),
                                res.size()));
  dbg(size, res.size());
  if (size <= res.size()) {
    res.truncate(size);
    return std::move(res);
  } else {
    return td::Status::Error("error while serializing a bag of cells: actual serialized size differs from estimated");
  }
}

td::Result<td::Ref<vm::DataCell>> CustomBagOfCells::deserialize_cell(int idx, td::Slice cell_slice,
                                                               td::Span<td::Ref<DataCell>> cells_span, BitReader& breader, CustomCellSerializationInfo& cell_info) {
  std::array<td::Ref<Cell>, 4> refs_buf;

  if (cell_info.end_offset != cell_slice.size()) {
    return td::Status::Error("unused space in cell serialization");
  }

  dbg(cell_info.refs_cnt);

  CellBuilder cb;
  TRY_RESULT(bits, cell_info.custom_get_bits(cell_slice));
  msg("Cell bits size = ", bits);
  // for (int i = 0; i < bits; ++i) {
  //   uint8_t bit = breader.read_bit();
  //   std::cerr << int(bit);
  //   unsigned char* ptr = static_cast<unsigned char*>(&bit);
  //   cb.store_bits(ptr, 1, 0);
  // }
  for (int i = 0; i < (bits + 7) / 8; ++i) {
    uint8_t byte = breader.read_bits(8);
    // std::cerr << "Loaded byte " << uint16_t(byte) << '\n';
    cb.store_bits(&byte, std::min(8, bits - i * 8));
  }
  // DCHECK(cell_info.data_offset == 2);
  // DCHECK(cell_info.refs_offset == cell_info.data_offset + (bits + 7) / 8);
  // DCHECK(cell_info.end_offset == cell_info.refs_offset + cell_info.refs_cnt * info.ref_bit_size);

  auto read_ref = [&]() -> uint64_t {
    return breader.read_bits(info.ref_bit_size);
  };

  auto refs = td::MutableSpan<td::Ref<Cell>>(refs_buf).substr(0, cell_info.refs_cnt);
  for (int k = 0; k < cell_info.refs_cnt; k++) {
    int ref_idx = idx + 1 + (int)read_ref();
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

  breader.flush_byte();

  return cell_info.custom_create_data_cell(cb, refs);
}

td::Result<long long> CustomBagOfCells::deserialize(const td::Slice& data, int max_roots) {
  msg("Running custom deserialize impl");
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
  for (int i = 0; i < info.root_count; i++) {
    int idx = 0;
    if (info.has_roots) {
      idx = (int)read_ref();
    }
    if (idx < 0 || idx >= info.cell_count) {
      return td::Status::Error(PSLICE() << "bag-of-cells invalid root index " << idx);
    }
    roots[i].idx = info.cell_count - idx - 1;
  }
  std::vector<Ref<DataCell>> cell_list;
  cell_list.reserve(cell_count);
  auto start_position = breader.flush_and_get_ptr();
  for (int i = 0; i < cell_count; i++) {
    CustomCellSerializationInfo cell_info;
    auto d1 = breader.read_bits(8);
    auto d2 = breader.read_bits(8);
    add_char("d1", d1);
    add_char("d2", d2);
    dbg(d1, d2);
    auto status = cell_info.custom_init(d1, d2, info.ref_bit_size);
    if (status.is_error()) {
      return td::Status::Error(PSLICE()
                                << "invalid bag-of-cells failed to deserialize cell #" << i << " " << status.error());
    }
    dbg(start_position, start_position + cell_info.end_offset);
    auto cell_slice = data.substr(start_position, cell_info.end_offset);
    dbg(cell_slice.size());
    // for (int i = 0; i < cell_slice.size(); ++i) {
    //   std::cerr << uint16_t(uint8_t(cell_slice[i])) << ' ';
    // }
    // std::cerr << '\n';
    start_position += cell_info.end_offset;
    // reconstruct cell with index cell_count - 1 - i
    int idx = cell_count - 1 - i;
    msg("Loading cell with idx ", idx);
    auto r_cell = deserialize_cell(idx, cell_slice, cell_list, breader, cell_info);
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
  dangle_count = info.absent_count;
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

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
      w.store_uint(bit_value, 1); // stores exactly byte
      bits = 0;
    }
  }
private:
  Writer& w;
  uint8_t bit_value;
  int bits;
};

std::vector<std::string> byte_code = {
"1100",
"01101",
"1101110",
"11010001",
"10110111",
"10101000",
"10110010",
"10011011",
"11111111",
"10110110",
"10100000",
"1001001",
"10110011",
"1010001",
"10000101",
"1011000",
"10111100",
"1010101",
"01001100",
"10100100",
"10011111",
"10000011",
"10100001",
"10110101",
"10101001",
"01000110",
"00100111",
"00000001",
"00111011",
"111111100",
"111100000",
"00110101",
"11010101",
"10110100",
"101110",
"10010101",
"00001111",
"111110110",
"00001010",
"00000111",
"011101",
"01011100",
"111101010",
"00101011",
"00011000",
"00000100",
"01100000",
"00111001",
"10101101",
"10001100",
"01111100",
"10011000",
"10010000",
"01000100",
"01111010",
"01001000",
"10001011",
"01010011",
"01001011",
"111100011",
"111101110",
"111000101",
"00100001",
"111110101",
"1000000",
"01010001",
"00101100",
"10011110",
"01110000",
"00001101",
"00001011",
"00001000",
"010110",
"00011010",
"111001101",
"111010000",
"01110001",
"111011101",
"111011011",
"111010100",
"10101111",
"01100010",
"10000010",
"01001010",
"00101001",
"01100011",
"01000101",
"111100001",
"01100001",
"00010010",
"00010100",
"111110111",
"00111111",
"10000110",
"111101111",
"111010110",
"10101100",
"10001001",
"01001101",
"00110000",
"01100101",
"00100101",
"01011101",
"111111000",
"11010011",
"01111000",
"00110010",
"110110001",
"00000000",
"00010011",
"111011010",
"10001101",
"10100111",
"10000111",
"10010100",
"01100111",
"10000100",
"111110001",
"00011101",
"111001001",
"00110001",
"111001011",
"111110100",
"00000010",
"111100101",
"110101101",
"111001110",
"111110010",
"11011001",
"01111101",
"10101110",
"01000011",
"10011010",
"01000001",
"00011111",
"111110011",
"11010000",
"00101111",
"00011110",
"111111010",
"01001001",
"00010101",
"00100000",
"111100010",
"10011100",
"111000111",
"111011000",
"00000101",
"110111111",
"111010010",
"111010101",
"00111100",
"00101010",
"00100110",
"111100111",
"111000000",
"01001110",
"111001111",
"00111000",
"110111101",
"0101011",
"10001110",
"01100110",
"10011001",
"01000111",
"00011001",
"00100100",
"00101000",
"10111101",
"00110110",
"01111011",
"00010001",
"01011110",
"01000010",
"111101001",
"111111101",
"01110011",
"10001010",
"00110011",
"110111110",
"00100011",
"00111010",
"110110111",
"110111100",
"01010000",
"01111001",
"01010101",
"00011100",
"00111101",
"01011111",
"10011101",
"111111011",
"1001011",
"10100101",
"10010001",
"01001111",
"00001001",
"00001110",
"00010110",
"00010111",
"10111111",
"10001111",
"00000011",
"01111110",
"00111110",
"111010011",
"01110010",
"01000000",
"10100110",
"00100010",
"111010001",
"111101100",
"00001100",
"111100100",
"111101011",
"00010000",
"111111001",
"111100110",
"111001000",
"111011001",
"110101111",
"110110000",
"111000110",
"111000001",
"10111110",
"01010010",
"01111111",
"111101101",
"111110000",
"111010111",
"111011100",
"111011110",
"11010010",
"110110101",
"00011011",
"00000110",
"111101000",
"00101101",
"111001100",
"111001010",
"00101110",
"111011111",
"00110100",
"110110110",
"00110111",
"110110100",
"110101001",
"111000010",
"01010100",
"110101110",
"01100100",
"111000100",
"111000011",
"110101000",
"110101100",
"10001000"
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
    return ans;
  }
  int flush_and_get_ptr() {
    flush_byte();
    return ptr;
  }
  void flush_byte() {
    if (bit_index != 0) {
      ++ptr;
      bit_index = 0;
    }
  }
private:
  const td::Slice& data;
  uint8_t ptr;
  int bit_index;
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
    int ref_byte_size;
    int offset_byte_size;
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
    long long parse_serialized_header(const td::Slice& slice);
    unsigned long long read_int(const unsigned char* ptr, unsigned bytes);
    unsigned long long read_ref(const unsigned char* ptr) {
      return read_int(ptr, ref_byte_size);
    }
    unsigned long long read_offset(const unsigned char* ptr) {
      return read_int(ptr, offset_byte_size);
    }
    void write_int(unsigned char* ptr, unsigned long long value, int bytes);
    void write_ref(unsigned char* ptr, unsigned long long value) {
      write_int(ptr, value, ref_byte_size);
    }
    void write_offset(unsigned char* ptr, unsigned long long value) {
      write_int(ptr, value, offset_byte_size);
    }
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
  td::Result<td::Ref<vm::DataCell>> deserialize_cell(int index, td::Slice data, td::Span<td::Ref<DataCell>> cells);
};

unsigned long long CustomBagOfCells::Info::read_int(const unsigned char* ptr, unsigned bytes) {
  unsigned long long res = 0;
  while (bytes > 0) {
    res = (res << 8) + *ptr++;
    --bytes;
  }
  return res;
}
void CustomBagOfCells::Info::write_int(unsigned char* ptr, unsigned long long value, int bytes) {
  ptr += bytes;
  while (bytes) {
    *--ptr = value & 0xff;
    value >>= 8;
    --bytes;
  }
  DCHECK(!bytes);
}

long long CustomBagOfCells::Info::parse_serialized_header(const td::Slice& slice) {
  invalidate();
  int sz = static_cast<int>(std::min(slice.size(), static_cast<std::size_t>(0xffff)));
  const unsigned char* ptr = slice.ubegin();
  // magic = (unsigned)read_int(ptr, 4);
  magic = boc_generic;
  has_crc32c = false;
  has_index = false;
  has_cache_bits = false;
  ref_byte_size = 0;
  offset_byte_size = 0;
  root_count = cell_count = absent_count = -1;
  index_offset = data_offset = data_size = total_size = 0;
  if (magic != boc_generic && magic != boc_idx && magic != boc_idx_crc32c) {
    magic = 0;
    return 0;
  }
  if (sz < 1) {
    return -10;
  }
  BitReader breader(slice, 0);
  td::uint8 byte = breader.read_bits(3);
  // td::uint8 byte = 2;
  ref_byte_size = byte & 7;
  if (ref_byte_size > 4 || ref_byte_size < 1) {
    return 0;
  }
  // if (sz < 6) {
  //   return -7 - 3 * ref_byte_size;
  // }
  // offset_byte_size = ptr[1];
  offset_byte_size = breader.read_bits(3);
  if (offset_byte_size > 8 || offset_byte_size < 1) {
    return 0;
  }
  int start_size = breader.flush_and_get_ptr();
  roots_offset = start_size + 3 * ref_byte_size + offset_byte_size;
  ptr += start_size;
  sz -= start_size;
  if (sz < ref_byte_size) {
    return -static_cast<int>(roots_offset);
  }
  cell_count = (int)read_ref(ptr);
  if (cell_count <= 0) {
    cell_count = -1;
    return 0;
  }
  if (sz < 2 * ref_byte_size) {
    return -static_cast<int>(roots_offset);
  }
  root_count = (int)read_ref(ptr + ref_byte_size);
  if (root_count <= 0) {
    root_count = -1;
    return 0;
  }
  index_offset = roots_offset;
  if (magic == boc_generic) {
    index_offset += (long long)root_count * ref_byte_size;
    has_roots = true;
  } else {
    if (root_count != 1) {
      return 0;
    }
  }
  data_offset = index_offset;
  if (sz < 3 * ref_byte_size) {
    return -static_cast<int>(roots_offset);
  }
  absent_count = (int)read_ref(ptr + 2 * ref_byte_size);
  if (absent_count < 0 || absent_count > cell_count) {
    return 0;
  }
  if (sz < 3 * ref_byte_size + offset_byte_size) {
    return -static_cast<int>(roots_offset);
  }
  data_size = read_offset(ptr + 3 * ref_byte_size);
  if (data_size > ((unsigned long long)cell_count << 10)) {
    return 0;
  }
  if (data_size > (1ull << 40)) {
    return 0;  // bag of cells with more than 1TiB data is unlikely
  }
  if (data_size < cell_count * (2ull + ref_byte_size) - ref_byte_size) {
    return 0;  // invalid header, too many cells for this amount of data bytes
  }
  valid = true;
  total_size = data_offset + data_size;
  return total_size;
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
  std::cerr << "Running custom serialize impl\n";
  auto store_ref = [&](unsigned long long value) { writer.store_uint(value, info.ref_byte_size); };
  auto store_offset = [&](unsigned long long value) { writer.store_uint(value, info.offset_byte_size); };

  td::uint8 byte{0};
  // 3, 4 - flags
  if (info.ref_byte_size < 1 || info.ref_byte_size > 7) {
    return 0;
  }
  byte |= static_cast<td::uint8>(info.ref_byte_size);
  bwriter.write_bits(byte, 3);

  bwriter.write_bits(info.offset_byte_size, 4);

  bwriter.flush_byte();
  store_ref(cell_count);
  store_ref(root_count);
  store_ref(0);
  store_offset(info.data_size);
  for (const auto& root_info : roots) {
    int k = cell_count - 1 - root_info.idx;
    DCHECK(k >= 0 && k < cell_count);
    store_ref(k);
  }
  DCHECK(writer.position() == info.index_offset);
  DCHECK((unsigned)cell_count == cell_list_.size());
  DCHECK(writer.position() == info.data_offset);
  size_t keep_position = writer.position();
  for (int i = 0; i < cell_count; ++i) {
    int idx = cell_count - 1 - i;
    // std::cerr << "Saving cell with idx " << idx << " with refnum " << int(cell_list_[idx].ref_num) << '\n';
    const auto& dc_info = cell_list_[idx];
    const Ref<DataCell>& dc = dc_info.dc_ref;
    unsigned char buf[256];
    int s = dc->serialize(buf, 256);
    writer.store_bytes(buf, s);
    DCHECK(dc->size_refs() == dc_info.ref_num);
    for (unsigned j = 0; j < dc_info.ref_num; ++j) {
      int k = cell_count - 1 - dc_info.ref_idx[j];
      DCHECK(k > i && k < cell_count);
      store_ref(k - i);
    }
  }
  writer.chk();
  DCHECK(writer.position() - keep_position == info.data_size);
  // DCHECK(writer.empty());
  return writer.position();
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
td::uint64 CustomBagOfCells::compute_sizes(int& r_size, int& o_size) {
  int rs = 0, os = 0;
  if (!root_count || !data_bytes) {
    r_size = o_size = 0;
    return 0;
  }
  while (cell_count >= (1LL << (rs << 3))) {
    rs++;
  }
  td::uint64 data_bytes_adj = data_bytes + (unsigned long long)int_refs * rs;
  td::uint64 max_offset = data_bytes_adj;
  while (max_offset >= (1ULL << (os << 3))) {
    os++;
  }
  if (rs > 4 || os > 8) {
    r_size = o_size = 0;
    return 0;
  }
  r_size = rs;
  o_size = os;
  return data_bytes_adj;
}

// Changes in this function may require corresponding changes in crypto/vm/large-boc-serializer.cpp
std::size_t CustomBagOfCells::estimate_serialized_size() {
  auto data_bytes_adj = compute_sizes(info.ref_byte_size, info.offset_byte_size);
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
  info.roots_offset = 0 + 1 + 0 + 3 * info.ref_byte_size + info.offset_byte_size;
  info.index_offset = info.roots_offset + info.root_count * info.ref_byte_size;
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
  if (size <= res.size()) {
    res.truncate(size);
    return std::move(res);
  } else {
    return td::Status::Error("error while serializing a bag of cells: actual serialized size differs from estimated");
  }
}

unsigned long long CustomBagOfCells::get_idx_entry_raw(int index) {
  if (index < 0) {
    return 0;
  }
  return custom_index.at(index);
}
bool CustomBagOfCells::get_cache_entry(int index) {
  return true;
}
unsigned long long CustomBagOfCells::get_idx_entry(int index) {
  auto raw = get_idx_entry_raw(index);
  return raw;
}

td::Result<td::Slice> CustomBagOfCells::get_cell_slice(int idx, td::Slice data) {
  unsigned long long offs = get_idx_entry(idx - 1);
  unsigned long long offs_end = get_idx_entry(idx);
  // std::cerr << "Offs " << offs << ' ' << offs_end << std::endl;
  if (offs > offs_end || offs_end > data.size()) {
    return td::Status::Error(PSLICE() << "invalid index entry [" << offs << "; " << offs_end << "], "
                                      << td::tag("data.size()", data.size()));
  }
  return data.substr(offs, td::narrow_cast<size_t>(offs_end - offs));
}

td::Result<td::Ref<vm::DataCell>> CustomBagOfCells::deserialize_cell(int idx, td::Slice cells_slice,
                                                               td::Span<td::Ref<DataCell>> cells_span) {
  TRY_RESULT(cell_slice, get_cell_slice(idx, cells_slice));
  std::array<td::Ref<Cell>, 4> refs_buf;

  CellSerializationInfo cell_info;
  // std::cerr << int(cell_slice[0]) << ' ' << uint32_t(cell_slice[1]) << '\n';
  TRY_STATUS(cell_info.init(cell_slice, info.ref_byte_size));
  if (cell_info.end_offset != cell_slice.size()) {
    return td::Status::Error("unused space in cell serialization");
  }

  auto refs = td::MutableSpan<td::Ref<Cell>>(refs_buf).substr(0, cell_info.refs_cnt);
  for (int k = 0; k < cell_info.refs_cnt; k++) {
    int ref_idx = idx + (int)info.read_ref(cell_slice.ubegin() + cell_info.refs_offset + k * info.ref_byte_size);
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

  return cell_info.create_data_cell(cell_slice, refs);
}

td::Result<long long> CustomBagOfCells::deserialize(const td::Slice& data, int max_roots) {
  std::cerr << "Running custom deserialize impl\n";
  get_og().clear();
  long long size_est = info.parse_serialized_header(data);
  if (size_est == 0) {
    return td::Status::Error(PSLICE() << "cannot deserialize bag-of-cells: invalid header, error " << size_est);
  }
  if (size_est < 0) {
    return size_est;
  }

  if (size_est > (long long)data.size()) {
    //LOG(ERROR) << "cannot deserialize bag-of-cells: not enough bytes (" << data.size() << " present, " << size_est
    //<< " required)";
    return -size_est;
  }
  //LOG(INFO) << "estimated size " << size_est << ", true size " << data.size();
  if (info.root_count > max_roots) {
    return td::Status::Error("Bag-of-cells has more root cells than expected");
  }

  cell_count = info.cell_count;
  roots.clear();
  roots.resize(info.root_count);
  auto* roots_ptr = data.substr(info.roots_offset).ubegin();
  for (int i = 0; i < info.root_count; i++) {
    int idx = 0;
    if (info.has_roots) {
      idx = (int)info.read_ref(roots_ptr + i * info.ref_byte_size);
    }
    if (idx < 0 || idx >= info.cell_count) {
      return td::Status::Error(PSLICE() << "bag-of-cells invalid root index " << idx);
    }
    roots[i].idx = info.cell_count - idx - 1;
  }
  {
    index_ptr = nullptr;
    unsigned long long cur = 0;
    custom_index.reserve(info.cell_count);

    auto cells_slice = data.substr(info.data_offset, info.data_size);

    for (int i = 0; i < info.cell_count; i++) {
      CellSerializationInfo cell_info;
      auto status = cell_info.init(cells_slice, info.ref_byte_size);
      if (status.is_error()) {
        return td::Status::Error(PSLICE()
                                  << "invalid bag-of-cells failed to deserialize cell #" << i << " " << status.error());
      }
      cells_slice = cells_slice.substr(cell_info.end_offset);
      cur += cell_info.end_offset;
      custom_index.push_back(cur);
    }
    if (!cells_slice.empty()) {
      return td::Status::Error(PSLICE() << "invalid bag-of-cells last cell #" << info.cell_count - 1 << ": end offset "
                                        << cur << " is different from total data size " << info.data_size);
    }
  }
  auto cells_slice = data.substr(info.data_offset, info.data_size);
  std::vector<Ref<DataCell>> cell_list;
  cell_list.reserve(cell_count);
  for (int i = 0; i < cell_count; i++) {
    // reconstruct cell with index cell_count - 1 - i
    int idx = cell_count - 1 - i;
    auto r_cell = deserialize_cell(idx, cells_slice, cell_list);
    if (r_cell.is_error()) {
      return td::Status::Error(PSLICE() << "invalid bag-of-cells failed to deserialize cell #" << idx << " "
                                        << r_cell.error());
    }
    cell_list.push_back(r_cell.move_as_ok());
    // std::cerr << "Loading cell with idx " << idx << " with refnum " << (*cell_list.back().get()).get_refs_cnt() << '\n';
    DCHECK(cell_list.back().not_null());
  }
  custom_index.clear();
  index_ptr = nullptr;
  root_count = info.root_count;
  dangle_count = info.absent_count;
  for (auto& root_info : roots) {
    root_info.cell = cell_list[root_info.idx];
  }
  cell_list.clear();
  return size_est;
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

constexpr bool use_lz4 = true;

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
  return use_lz4 ? td::lz4_compress(with_bwt) : std::move(with_bwt);
}

td::BufferSlice decompress(td::Slice data) {
  auto decompressed = use_lz4 ? td::lz4_decompress(data, 2 << 20).move_as_ok() : td::BufferSlice(data);
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

  // freopen("res/output.txt", "a", stderr);

  // std::vector<std::pair<int, int>> stcnt(vm::cnt.begin(), vm::cnt.end());
  // std::sort(stcnt.begin(), stcnt.end(), [](auto a, auto b) {
  //   return a.second > b.second;
  // });
  // for (auto [v, c] : stcnt) {
  //   std::cerr << v << ' ' << c << std::endl;
  // }
}

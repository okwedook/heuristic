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
#include "td/utils/lz4.h"
#include "td/utils/base64.h"
#include "vm/boc.h"
#include "block/block-auto.h"
#include "crypto/vm/boc-writers.h"
#include "tdutils/td/utils/misc.h"

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
  td::Result<td::Ref<vm::DataCell>> deserialize_cell(int index, td::Slice data, td::Span<td::Ref<DataCell>> cells,
                                                     std::vector<td::uint8>* cell_should_cache);
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
  td::uint8 byte = ptr[0];
  // td::uint8 byte = 2;
  if (magic == boc_generic) {
    has_index = (byte >> 7) % 2 == 1;
    has_crc32c = (byte >> 6) % 2 == 1;
    has_cache_bits = (byte >> 5) % 2 == 1;
  } else {
    has_index = true;
    has_crc32c = magic == boc_idx_crc32c;
  }
  if (has_cache_bits && !has_index) {
    return 0;
  }
  ref_byte_size = byte & 7;
  if (ref_byte_size > 4 || ref_byte_size < 1) {
    return 0;
  }
  // if (sz < 6) {
  //   return -7 - 3 * ref_byte_size;
  // }
  offset_byte_size = ptr[1];
  if (offset_byte_size > 8 || offset_byte_size < 1) {
    return 0;
  }
  roots_offset = 2 + 3 * ref_byte_size + offset_byte_size;
  ptr += 2;
  sz -= 2;
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
  if (has_index) {
    data_offset += (long long)cell_count * offset_byte_size;
  }
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
  total_size = data_offset + data_size + (has_crc32c ? 4 : 0);
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
  std::cerr << "Running custom serialize impl\n";
  auto store_ref = [&](unsigned long long value) { writer.store_uint(value, info.ref_byte_size); };
  auto store_offset = [&](unsigned long long value) { writer.store_uint(value, info.offset_byte_size); };

  // writer.store_uint(info.magic, 4);

  td::uint8 byte{0};
  // 3, 4 - flags
  if (info.ref_byte_size < 1 || info.ref_byte_size > 7) {
    return 0;
  }
  byte |= static_cast<td::uint8>(info.ref_byte_size);
  writer.store_uint(byte, 1);

  writer.store_uint(info.offset_byte_size, 1);
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
  if (info.has_index) {
    std::size_t offs = 0;
    for (int i = cell_count - 1; i >= 0; --i) {
      const Ref<DataCell>& dc = cell_list_[i].dc_ref;
      offs += dc->get_serialized_size() + dc->size_refs() * info.ref_byte_size;
      auto fixed_offset = offs;
      if (info.has_cache_bits) {
        fixed_offset = offs * 2 + cell_list_[i].should_cache;
      }
      store_offset(fixed_offset);
    }
    DCHECK(offs == info.data_size);
  }
  DCHECK(writer.position() == info.data_offset);
  size_t keep_position = writer.position();
  for (int i = 0; i < cell_count; ++i) {
    const auto& dc_info = cell_list_[cell_count - 1 - i];
    const Ref<DataCell>& dc = dc_info.dc_ref;
    unsigned char buf[256];
    int s = dc->serialize(buf, 256);
    writer.store_bytes(buf, s);
    DCHECK(dc->size_refs() == dc_info.ref_num);
    for (unsigned j = 0; j < dc_info.ref_num; ++j) {
      int k = cell_count - 1 - dc_info.ref_idx[j];
      DCHECK(k > i && k < cell_count);
      store_ref(k);
    }
  }
  writer.chk();
  DCHECK(writer.position() - keep_position == info.data_size);
  DCHECK(writer.remaining() == (info.has_crc32c ? 4 : 0));
  if (info.has_crc32c) {
    unsigned crc = writer.get_crc32();
    writer.store_uint(td::bswap32(crc), 4);
  }
  DCHECK(writer.empty());
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
  int crc_size = info.has_crc32c ? 4 : 0;
  info.roots_offset = 0 + 1 + 1 + 3 * info.ref_byte_size + info.offset_byte_size;
  info.index_offset = info.roots_offset + info.root_count * info.ref_byte_size;
  info.data_offset = info.index_offset;
  if (info.has_index) {
    info.data_offset += (long long)cell_count * info.offset_byte_size;
  }
  info.magic = Info::boc_generic;
  info.data_size = data_bytes_adj;
  info.total_size = info.data_offset + data_bytes_adj + crc_size;
  auto res = td::narrow_cast_safe<size_t>(info.total_size);
  if (res.is_error()) {
    return 0;
  }
  return res.ok();
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
  if (size == res.size()) {
    return std::move(res);
  } else {
    return td::Status::Error("error while serializing a bag of cells: actual serialized size differs from estimated");
  }
}

unsigned long long CustomBagOfCells::get_idx_entry_raw(int index) {
  if (index < 0) {
    return 0;
  }
  if (!info.has_index) {
    return custom_index.at(index);
  } else if (index < info.cell_count && index_ptr) {
    return info.read_offset(index_ptr + (long)index * info.offset_byte_size);
  } else {
    // throw ?
    return 0;
  }
}
bool CustomBagOfCells::get_cache_entry(int index) {
  if (!info.has_cache_bits) {
    return true;
  }
  if (!info.has_index) {
    return true;
  }
  auto raw = get_idx_entry_raw(index);
  return raw % 2 == 1;
}
unsigned long long CustomBagOfCells::get_idx_entry(int index) {
  auto raw = get_idx_entry_raw(index);
  if (info.has_cache_bits) {
    raw /= 2;
  }
  return raw;
}

td::Result<td::Slice> CustomBagOfCells::get_cell_slice(int idx, td::Slice data) {
  unsigned long long offs = get_idx_entry(idx - 1);
  unsigned long long offs_end = get_idx_entry(idx);
  if (offs > offs_end || offs_end > data.size()) {
    return td::Status::Error(PSLICE() << "invalid index entry [" << offs << "; " << offs_end << "], "
                                      << td::tag("data.size()", data.size()));
  }
  return data.substr(offs, td::narrow_cast<size_t>(offs_end - offs));
}

td::Result<td::Ref<vm::DataCell>> CustomBagOfCells::deserialize_cell(int idx, td::Slice cells_slice,
                                                               td::Span<td::Ref<DataCell>> cells_span,
                                                               std::vector<td::uint8>* cell_should_cache) {
  TRY_RESULT(cell_slice, get_cell_slice(idx, cells_slice));
  std::array<td::Ref<Cell>, 4> refs_buf;

  CellSerializationInfo cell_info;
  TRY_STATUS(cell_info.init(cell_slice, info.ref_byte_size));
  if (cell_info.end_offset != cell_slice.size()) {
    return td::Status::Error("unused space in cell serialization");
  }

  auto refs = td::MutableSpan<td::Ref<Cell>>(refs_buf).substr(0, cell_info.refs_cnt);
  for (int k = 0; k < cell_info.refs_cnt; k++) {
    int ref_idx = (int)info.read_ref(cell_slice.ubegin() + cell_info.refs_offset + k * info.ref_byte_size);
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
    if (cell_should_cache) {
      auto& cnt = (*cell_should_cache)[ref_idx];
      if (cnt < 2) {
        cnt++;
      }
    }
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
  if (info.has_crc32c) {
    unsigned crc_computed = td::crc32c(td::Slice{data.ubegin(), data.uend() - 4});
    unsigned crc_stored = td::as<unsigned>(data.uend() - 4);
    if (crc_computed != crc_stored) {
      return td::Status::Error(PSLICE() << "bag-of-cells CRC32C mismatch: expected " << td::format::as_hex(crc_computed)
                                        << ", found " << td::format::as_hex(crc_stored));
    }
  }

  cell_count = info.cell_count;
  std::vector<td::uint8> cell_should_cache;
  if (info.has_cache_bits) {
    cell_should_cache.resize(cell_count, 0);
  }
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
    if (info.has_cache_bits) {
      auto& cnt = cell_should_cache[idx];
      if (cnt < 2) {
        cnt++;
      }
    }
  }
  if (info.has_index) {
    index_ptr = data.substr(info.index_offset).ubegin();
    // TODO: should we validate index here
  } else {
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
  std::array<td::Ref<Cell>, 4> refs_buf;
  for (int i = 0; i < cell_count; i++) {
    // reconstruct cell with index cell_count - 1 - i
    int idx = cell_count - 1 - i;
    auto r_cell = deserialize_cell(idx, cells_slice, cell_list, info.has_cache_bits ? &cell_should_cache : nullptr);
    if (r_cell.is_error()) {
      return td::Status::Error(PSLICE() << "invalid bag-of-cells failed to deserialize cell #" << idx << " "
                                        << r_cell.error());
    }
    cell_list.push_back(r_cell.move_as_ok());
    DCHECK(cell_list.back().not_null());
  }
  if (info.has_cache_bits) {
    for (int idx = 0; idx < cell_count; idx++) {
      auto should_cache = cell_should_cache[idx] > 1;
      auto stored_should_cache = get_cache_entry(idx);
      if (should_cache != stored_should_cache) {
        return td::Status::Error(PSLICE() << "invalid bag-of-cells cell #" << idx << " has wrong cache flag "
                                          << stored_should_cache);
      }
    }
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

td::BufferSlice compress(td::Slice data) {
  td::Ref<vm::Cell> root = vm::std_boc_deserialize(data).move_as_ok();
  td::BufferSlice serialized = vm::custom_boc_serialize(root).move_as_ok();
  return use_lz4 ? td::lz4_compress(serialized) : std::move(serialized);
}

td::BufferSlice decompress(td::Slice data) {
  vm::Ref<vm::Cell> root;
  if (use_lz4) {
    td::BufferSlice serialized = td::lz4_decompress(data, 2 << 20).move_as_ok();
    root = vm::custom_boc_deserialize(serialized).move_as_ok();
  } else {
    root = vm::custom_boc_deserialize(data).move_as_ok();
  }
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

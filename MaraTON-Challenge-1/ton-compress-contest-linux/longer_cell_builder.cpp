/*
 * 01 - Cells
 *
 * This file provides examples for working with cells.
 * A cell is an object that stores up to 1023 bits of data and up to 4 references to other cells.
 * Cell builder is an object used to build new cells.
 * Cell slice is an object used to read data from cells.
 *
 * You can be familiar with cells, cell slices and cell builders if you participated in FunC contests.
 */

#include <iostream>
#include "vm/cells/CellBuilder.h"
#include "vm/cells/CellSlice.h"
#include "vm/cellslice.h"
#include "td/utils/buffer.h"      // td::BufferSlice, td::Slice, td::MutableSlice
#include "td/utils/misc.h"        // td::buffer_to_hex
#include "common/util.h"          // td::base64_decode, td::str_base64_encode
#include "vm/boc.h"               // vm::std_boc_serialize, ...
#include "vm/cellslice.h"         // vm::load_cell_slice
#include "td/utils/lz4.h"         // td::lz4_compress, td::lz4_decompress

int main() {

  vm::CellBuilder cb_spec;
  cb_spec.store_long(2, 8);
  cb_spec.store_long(0x1234567812345678LL, 64);
  cb_spec.store_long(0x1234567812345678LL, 64);
  cb_spec.store_long(0x1234567812345678LL, 64);
  cb_spec.store_long(0x1234567812345678LL, 64);
  td::Ref<vm::Cell> cell_spec = cb_spec.finalize(true);  // Create a special cell

  bool is_special = false;
  vm::CellSlice cs_spec = vm::load_cell_slice_special(cell_spec, is_special);
  cs_spec.print_rec(std::cout);

  // Creating a new cell
  vm::CellBuilder cb;
  cb.store_long(100, 64);
  unsigned char* ptr = new unsigned char[5];
  ptr[0] = 10;
  ptr[1] = 3;
  cb.store_bits(ptr, 11);
  cb.store_ref(cell_spec);
  td::Ref<vm::Cell> cell = cb.finalize();  // Create a cell from cell builder

  vm::CellSlice cs = vm::load_cell_slice(cell);
  cs.print_rec(std::cout);

  cb.reset();
  cb.store_long(123, 32);
  cb.store_long(456, 32);
  ptr[0] = 11;
  cb.store_bits(ptr, 5);
  cb.store_ref(cell);
  cb.store_ref(cell_spec);
  td::Ref<vm::Cell> cell2 = cb.finalize();  // Create a cell from cell builder
  

  td::BufferSlice serialized_31 = vm::std_boc_serialize(cell2, 31).move_as_ok();
  auto base_64 = td::str_base64_encode(serialized_31);

  std::cout << base_64 << '\n';

  return 0;
}

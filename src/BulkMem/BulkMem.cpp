
#include "BulkMem/BulkMem.hpp"

using namespace slf;

i_t MemSize::TOTAL_SIZE = 0;
i_t MemSize::MAX_SIZE   = 0;

MemSize::MemSize()
{

}

MemSize::~MemSize()
{

}

void MemSize::show_memory_usage(void)
{
    std::cout << "MemSize summary: TOTAL_SIZE = " 
              << 1.0 * TOTAL_SIZE / GB 
              << " GB, MAX_SIZE = " 
              << 1.0 * MAX_SIZE / GB << " GB." << std::endl;
}

/*
 * BulkMem.hpp
 *
 *  Created on: Aug 7, 2018
 *      Author: yyhu
 */

#ifndef INCLUDE_BULKMEM_BULKMEM_HPP_
#define INCLUDE_BULKMEM_BULKMEM_HPP_

#include <iostream>
#include <string>
#include <sstream>

#include "SLFException/SLFException.hpp"

namespace slf
{

typedef unsigned long i_t;

class MemSize
{
public:
    MemSize();
    ~MemSize();

    static void show_memory_usage(void);

protected:
    static const i_t GB = 1024 * 1024 * 1024;
    static i_t TOTAL_SIZE;
    static i_t MAX_SIZE;
};

template<class _T>
class BulkMem : public MemSize
{

public:
    BulkMem(i_t* dim, i_t nDim = 1, bool faked = false);
    ~BulkMem();

    _T* get(void);
    i_t get_n_dim(void);
    i_t* get_dim(void);
    i_t get_size(void);

    bool is_faked(void);

    i_t total_size(void);

    void assign(const _T& val);
    void assign(const _T* val);

public:
    _T* ptr;

protected:
    i_t mSize;
    i_t mNDim;
    i_t* mDim;
    bool mFaked;

private:
    BulkMem(const BulkMem& bm) { }    
    BulkMem(BulkMem& bm) { }    
};

template<typename _T>
BulkMem<_T>::BulkMem(i_t* dim, i_t nDim, bool faked)
: MemSize()
{
    mNDim = nDim;
    mDim  = new i_t[mNDim];
    mSize = 1;

    for ( i_t i = 0; i < mNDim; ++i )
    {
        mSize  *= dim[i];
        mDim[i] = dim[i];
    }

    mFaked = faked;
    if ( false == mFaked )
    {
        ptr = new _T[ mSize ];
    }

    // Update TOTAL_SIZE.
    TOTAL_SIZE += mSize * sizeof(_T);

    if ( TOTAL_SIZE > MAX_SIZE )
    {
        MAX_SIZE = TOTAL_SIZE;
    }
}

template<typename _T> 
BulkMem< _T >::~BulkMem()
{
    if ( ptr != NULL && false == mFaked )
    {
        delete [] ptr; ptr = NULL;
    }

    if ( mDim != NULL )
    {
        delete [] mDim; mDim = NULL;
    }

    // Update TOTAL_SIZE;
    TOTAL_SIZE -= mSize * sizeof(_T);
}

template<typename _T>
_T* BulkMem<_T>::get(void)
{
    return ptr;
}

template<typename _T>
i_t BulkMem<_T>::get_n_dim(void)
{
    return mNDim;
}

template<typename _T>
i_t* BulkMem<_T>::get_dim(void)
{
    return mDim;
}

template<typename _T>
i_t BulkMem<_T>::get_size(void)
{
    return mSize;
}

template<typename _T>
bool BulkMem<_T>::is_faked(void)
{
    return mFaked;
}

template<typename _T>
i_t BulkMem<_T>::total_size(void)
{
    return TOTAL_SIZE;
}

template<typename _T>
void BulkMem<_T>::assign(const _T& val)
{
    for ( i_t i = 0; i < mSize; ++i )
    {
        ptr[i] = val;
    }
}

template<typename _T>
void BulkMem<_T>::assign(const _T* val)
{
    for ( i_t i = 0; i < mSize; ++i )
    {
        ptr[i] = val[i];
    }
}

} // Namespace slf.

#endif /* INCLUDE_BULKMEM_BULKMEM_HPP_ */

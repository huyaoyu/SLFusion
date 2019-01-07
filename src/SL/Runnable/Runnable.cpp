/*
 * Runnable.cpp
 *
 *  Created on: Jun 24, 2018
 *      Author: yyhu
 */

#include "Runnable/Runnable.hpp"

using namespace slf;

Runnable::Runnable(std::string& name)
: mName(name)
{

}

Runnable::Runnable(const std::string& name)
: mName(name)
{

}

Runnable::~Runnable(void)
{

}

void Runnable::set_name(std::string& name)
{
	mName = name;
}

std::string& Runnable::get_name(void)
{
	return mName;
}

void Runnable::show_header(void)
{
	std::cout << "=== " << mName << " ===" << std::endl << std::endl;
}

void Runnable::show_footer(void)
{
	std::cout <<"=== End of " << mName << " ===" << std::endl << std::endl;
}

Runnable::RES_t Runnable::run(void)
{
	return Runnable::OK;
}

void Runnable::show_size_of_types(void)
{
	std::cout << "The size of types: " << std::endl;

	std::cout << "sizeof(char)      = " << sizeof(char) << std::endl;
	std::cout << "sizeof(uchar)     = " << sizeof(unsigned char) << std::endl;
	// std::cout << "(ucahr)(1) - (uchar)(255) = " << (unsigned char)( (unsigned char)(1) - (unsigned char)(255) ) << std::endl;
	std::cout << "sizeof(int)       = " << sizeof(int) << std::endl;
	std::cout << "sizeof(uint)      = " << sizeof(uint) << std::endl;
	std::cout << "sizeof(short)     = " << sizeof(short) << std::endl;
	std::cout << "sizeof(ushort)    = " << sizeof(ushort) << std::endl;
	std::cout << "sizeof(long)      = " << sizeof(long) << std::endl;
	std::cout << "sizeof(ulong)     = " << sizeof(unsigned long) << std::endl;
	std::cout << "sizeof(long long) = " << sizeof(long long) << std::endl;
	std::cout << "sizeof(float)     = " << sizeof(float) << std::endl;
	std::cout << "sizeof(double)    = " << sizeof(double) << std::endl;
}

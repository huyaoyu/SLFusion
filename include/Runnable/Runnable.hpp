/*
 * Runnable.hpp
 *
 *  Created on: Jun 24, 2018
 *      Author: yyhu
 */

#ifndef INCLUDE_RUNNABLE_HPP_
#define INCLUDE_RUNNABLE_HPP_

#include <iostream>
#include <string>
#include <sstream>

#include "SLFException/SLFException.hpp"
#include "BulkMem/BulkMem.hpp"

namespace slf
{

class Runnable
{
public:
	typedef enum
	{
		OK = 0,
		ERROR = -1
	}RES_t;

public:
	Runnable(std::string& name);
	Runnable(const std::string& name);
	virtual ~Runnable(void);

	virtual Runnable::RES_t run(void);

	void set_name(std::string& name);
	std::string& get_name(void);

	static void show_size_of_types(void);

protected:
	void show_header(void);
	void show_footer(void);

protected:
	std::string mName; // The name of this Runnable object.
};

}

#endif /* INCLUDE_RUNNABLE_HPP_ */

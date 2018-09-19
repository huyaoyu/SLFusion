/*
 * VC_LLF.hpp
 *
 *  Created on: Jul 6, 2018
 *      Author: yyhu
 */

#ifndef INCLUDE_VIGNETTINGCORRECTION_VC_LLF_HPP_
#define INCLUDE_VIGNETTINGCORRECTION_VC_LLF_HPP_

#include "Runnable/Runnable.hpp"

namespace slf
{

class VC_LLF : public Runnable
{
public:
	typedef float real;

public:
	VC_LLF();
	~VC_LLF();

	Runnable::RES_t run(void);
};

}


#endif /* INCLUDE_VIGNETTINGCORRECTION_VC_LLF_HPP_ */

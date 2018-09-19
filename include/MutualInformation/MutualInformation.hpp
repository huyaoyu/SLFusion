/*
 * MutualInformation.hpp
 *
 *  Created on: Jun 28, 2018
 *      Author: yyhu
 */

#ifndef INCLUDE_MUTUALINFORMATION_MUTUALINFORMATION_HPP_
#define INCLUDE_MUTUALINFORMATION_MUTUALINFORMATION_HPP_

#include "Runnable/Runnable.hpp"

namespace slf
{

class Run_MutualInformation : public Runnable
{
public:
	typedef float real;

public:
	Run_MutualInformation();
	~Run_MutualInformation();

	Runnable::RES_t run(void);
};

}



#endif /* INCLUDE_MUTUALINFORMATION_MUTUALINFORMATION_HPP_ */

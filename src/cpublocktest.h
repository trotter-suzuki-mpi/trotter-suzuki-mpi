#ifndef __CPUBLOCKTEST_H
#define __CPUBLOCKTEST_H

#include <cppunit/extensions/HelperMacros.h>

class CPUBlockTest: public CppUnit::TestFixture{
	CPPUNIT_TEST_SUITE( CPUBlockTest );
	CPPUNIT_TEST( test_block_kernel_vertical );
	CPPUNIT_TEST( test_block_kernel_horizontal );
	CPPUNIT_TEST_SUITE_END(); 
	
	public:
		void setUp();
		void tearDown();
		void test_block_kernel_vertical();
		void test_block_kernel_horizontal();
};

class Matrix{
	public:
		Matrix(float *matrix_real, float *matrix_imag, int width, int height);
		bool operator ==(const Matrix &other) const;
		
	private:
		float *m_real;
		float *m_imag;
		int m_width, m_height;
};
#endif

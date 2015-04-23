#include "cpublocktest.h"
#include "cpublock.h"
#include "common.h"
#include <iostream>

CPPUNIT_TEST_SUITE_REGISTRATION( CPUBlockTest );

void CPUBlockTest::setUp(){}
void CPUBlockTest::tearDown(){}

void CPUBlockTest::test_block_kernel_vertical(){
	//Set Up
	int DIM=640;
	int offset=0;
	float a=h_a, b=h_b;
	
	float *block_real = new float[DIM*DIM];
    float *block_imag = new float[DIM*DIM];
    float *block_real_expected = new float[DIM*DIM];
    float *block_imag_expected = new float[DIM*DIM];
    
    //initialize block_real, block_imag
    for(int i=0; i<DIM; i++){
		for(int j=0; j<DIM; j++){
			block_real[i*DIM+j] = 1.;
			block_imag[i*DIM+j] = 0.;
		}
	}
	
	//inizialize block_real_expected, block_imag_expected
	for(int i=0; i<DIM; i++){
		for(int j=0; j<DIM; j++){
			if((i==0 || i==DIM-1) && ((j+offset+1)%2)==0){
				block_real_expected[i*DIM+j] = 1.;
				block_imag_expected[i*DIM+j] = 0.;
			}
			else{
				block_real_expected[i*DIM+j] = a;
				block_imag_expected[i*DIM+j] = b;
			}
		}
	}

	//Process block_real, block_imag
	block_kernel_vertical(offset, DIM, DIM, DIM, a , b, block_real, block_imag);
	
	Matrix matrix_processed(block_real, block_imag, DIM, DIM);
	Matrix matrix_expected(block_real_expected, block_imag_expected, DIM, DIM);
	
	//Check
	CPPUNIT_ASSERT( matrix_processed == matrix_expected );
	std::cout << "TEST FUNCTION: block_kernel_vertical " << std::endl;
}

void CPUBlockTest::test_block_kernel_horizontal(){
	//Set Up
	int DIM=640;
	int offset=0;
	float a=h_a, b=h_b;
	float *block_real = new float[DIM*DIM];
    float *block_imag = new float[DIM*DIM];
    float *block_real_expected = new float[DIM*DIM];
    float *block_imag_expected = new float[DIM*DIM];
    
    //initialize block_real, block_imag
    for(int i=0; i<DIM; i++){
		for(int j=0; j<DIM; j++){
			block_real[i*DIM+j] = 1.;
			block_imag[i*DIM+j] = 0.;
		}
	}
	
	//inizialize block_real_expected, block_imag_expected
	for(int i=0; i<DIM; i++){
		for(int j=0; j<DIM; j++){
			if((j==0 || j==DIM-1) && ((i+offset+1)%2)==0){
				block_real_expected[i*DIM+j] = 1.;
				block_imag_expected[i*DIM+j] = 0.;
			}
			else{
				block_real_expected[i*DIM+j] = a;
				block_imag_expected[i*DIM+j] = b;
			}
		}
	}

	//Process block_real, block_imag
	block_kernel_horizontal(offset, DIM, DIM, DIM, a , b, block_real, block_imag);
	
	Matrix matrix_processed(block_real, block_imag, DIM, DIM);
	Matrix matrix_expected(block_real_expected, block_imag_expected, DIM, DIM);
	
	//Check
	CPPUNIT_ASSERT( matrix_processed == matrix_expected );
	std::cout << "TEST FUNCTION: block_kernel_horizontal " << std::endl;
}

//Members of class Matrix
Matrix::Matrix(float *matrix_real, float *matrix_imag, int width, int height){
	m_real = new float[width*height];
	m_imag = new float[width*height];
	m_width = width;
	m_height = height;
	
	for(int i=0; i<m_height; i++){
		for(int j=0; j<m_width; j++){
			m_real[i*m_width+j] = matrix_real[i*m_width+j];
			m_imag[i*m_width+j] = matrix_imag[i*m_width+j];
		}
	}
}

bool Matrix::operator ==(const Matrix &other) const {
	bool var=false;
	if((m_height == other.m_height) && (m_width == other.m_width)){
		var = true;
		for(int i=0; i<m_height; i++){
			for(int j=0; j<m_width; j++){
				if((m_real[i*m_width+j] != other.m_real[i*m_width+j]) || (m_imag[i*m_width+j] != other.m_imag[i*m_width+j]))
					var = false;
			}
		}
	}
	return var;
}
	

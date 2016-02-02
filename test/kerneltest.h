#ifndef __CPUKERNELTEST_H
#define __CPUKERNELTEST_H

#include <string>
#include <cppunit/extensions/HelperMacros.h>
#include "trottersuzuki.h"

#define TOLERANCE 1.e-3
#define NORM_TOLERANCE 1.e-5

class KernelTest: public CppUnit::TestFixture {
public:
    std::string kernel_type;
};

class CpuKernelTest: public KernelTest {
public:
    void setUp();
};


template<class F>
class my_test: public F {
    CPPUNIT_TEST_SUITE(my_test<F>);
    CPPUNIT_TEST( free_particle_test );
    CPPUNIT_TEST( harmonic_oscillator_test );
    CPPUNIT_TEST( imaginary_harmonic_oscillator_test );
    CPPUNIT_TEST( intra_particle_interaction_test );
    CPPUNIT_TEST( imaginary_intra_particle_interaction_test );
    CPPUNIT_TEST( rotating_frame_of_reference_test );
    CPPUNIT_TEST( imaginary_rotating_frame_of_reference_test );
    CPPUNIT_TEST( mixed_BEC_test );
    CPPUNIT_TEST( imaginary_mixed_BEC_test );
    CPPUNIT_TEST_SUITE_END();

    void free_particle_test();
    void harmonic_oscillator_test();
    void imaginary_harmonic_oscillator_test();
    void intra_particle_interaction_test();
    void imaginary_intra_particle_interaction_test();
    void rotating_frame_of_reference_test();
    void imaginary_rotating_frame_of_reference_test();
    void mixed_BEC_test();
    void imaginary_mixed_BEC_test();
};

CPPUNIT_TEST_SUITE_REGISTRATION(my_test<CpuKernelTest>);
#ifdef CUDA
class GpuKernelTest: public KernelTest {
public:
    void setUp();
};
class HybridKernelTest: public KernelTest {
public:
    void setUp();
};
CPPUNIT_TEST_SUITE_REGISTRATION(my_test<GpuKernelTest>);
CPPUNIT_TEST_SUITE_REGISTRATION(my_test<HybridKernelTest>);
#endif

#endif

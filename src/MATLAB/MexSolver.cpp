#include "mex.h"
#include "trottersuzuki.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[]) {
    
    int nCol = mxGetN(prhs[0]);
    int nRow = mxGetM(prhs[0]);
    
    //check inputs           
    if(nrhs != 12) {
		mexErrMsgIdAndTxt("MexTrotter:"," twelve inputs required.");
	}
	
	if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 1: notDouble Input, matrix must be type double.");
	}
	
	if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 2: notDouble Input, matrix must be type double.");
	}
	
	if( !mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 3: notScalar Input, particle mass must be a scalar.");
	}
	
	if( !mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 4: notScalar Input, coupling constant must be a scalar.");
	}
	
	if( !mxIsDouble(prhs[4]) || mxIsComplex(prhs[4])) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 5: notDouble Input, external potential matrix must be type double.");
	}
	
	if( !mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetNumberOfElements(prhs[5]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 6: notScalar Input, delta_x must be a scalar.");
	}
	if( !mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetNumberOfElements(prhs[6]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 7: notScalar Input, delta_y must be a scalar.");
	}
	if( !mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || mxGetNumberOfElements(prhs[7]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 8: notScalar Input, delta_t must be a scalar.");
	}
	if( mxIsComplex(prhs[8]) || mxGetNumberOfElements(prhs[8]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 9: notScalar Input, number of iterations must be a scalar.");
	}
	if( mxIsComplex(prhs[9]) || mxGetNumberOfElements(prhs[9]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 10: notScalar Input, kernel type must be a scalar.");
	}
	
	if( mxIsComplex(prhs[11]) || mxGetNumberOfElements(prhs[11]) != 1 ) {
		mexErrMsgIdAndTxt("MexTrotter:"," input 12: notScalar Input, imag time must be a scalar.");
	}
	
	//check matrices dimensions    
    if (nCol != mxGetN(prhs[1])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Input 2 must have the same number of columns as input 1");
    }
    if (nRow != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Input 2 must have the same number of rows as input 1");
    }
	
	if (nCol != mxGetN(prhs[4])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Input 5 must have the same number of columns as input 1");
    }
    if (nRow != mxGetM(prhs[4])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Input 5 must have the same number of rows as input 1");
    }
    
	//check outputs
	if(nlhs != 2) {
		mexErrMsgIdAndTxt("MexTrotter:"," two outputs required.");
	}
	/*
	if (nCol != mxGetN(plhs[0])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Output 1 must have the same number of columns as input 1");
    }
    if (nRow != mxGetM(plhs[0])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Output 1 must have the same number of rows as input 1");
    }
    
    if (nCol != mxGetN(plhs[1])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Output 2 must have the same number of columns as input 1");
    }
    if (nRow != mxGetM(plhs[1])) {
        mexErrMsgIdAndTxt("MexTrotter:", "Output 2 must have the same number of rows as input 1");
    }
    */
    // initialize matrices
    double *Mp_real = mxGetPr(prhs[0]);
    double *p_real = new double[nCol * nRow];
    double *Mp_imag = mxGetPr(prhs[1]);
    double *p_imag = new double[nCol * nRow];
    double *Mext_pot = mxGetPr(prhs[4]);
    double *ext_pot = new double[nCol * nRow];
    int *Mperiods = (int*)mxGetPr(prhs[10]);
    int *periods = new int[2];
    
    for (int i = 0; i < nRow; i++) {
        for (int j = 0; j < nCol; j++) {
            p_real[i * nCol + j] = (double)Mp_real[j * nRow + i];
            p_imag[i * nCol + j] = (double)Mp_imag[j * nRow + i];
            ext_pot[i * nCol + j] = (double)Mext_pot[j * nRow + i];
        }
    }
    for (int i = 0; i < 2; i++) {
        periods[i] = (int)Mperiods[i];
    }
    
    //initialize scalars
    double particle_mass = (double)mxGetPr(prhs[2])[0];
    double coupling_const = (double)mxGetPr(prhs[3])[0];
    double delta_x = (double)mxGetPr(prhs[5])[0];
    double delta_y = (double)mxGetPr(prhs[6])[0];
    double delta_t = (double)mxGetPr(prhs[7])[0];
    int iterations = (int)mxGetPr(prhs[8])[0];
    char* kernel_type_c = mxArrayToString(prhs[9]);
    string kernel_type;
    if(kernel_type_c != NULL) {
        kernel_type = kernel_type_c;
    }
    else {
        kernel_type = "";
    }
    mxFree(kernel_type_c);
    bool imag_time = (bool)mxGetPr(prhs[11])[0];
    
    //launch the solver
    solver(p_real, p_imag, particle_mass, coupling_const, ext_pot, nCol, nRow, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time);
    
    //Set output
    plhs[0] = mxCreateDoubleMatrix(nRow, nCol, mxREAL);
    double* Op_real = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(nRow, nCol, mxREAL);
    double* Op_imag = mxGetPr(plhs[1]);
    for (int i = 0; i < nRow; i++) {
        for (int j = 0; j < nCol; j++) {
            Op_real[j * nRow + i] = (double)p_real[i * nCol + j];
            Op_imag[j * nRow + i] = (double)p_imag[i * nCol + j];
        }
    }
}

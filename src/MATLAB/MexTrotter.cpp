
#include "mex.h"
#include "trotter.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	double h_a = (double)mxGetPr(prhs[0])[0];
	double h_b = (double)mxGetPr(prhs[1])[0];
	int nCol = mxGetN(prhs[2]);
	int nRow = mxGetM(prhs[2]);
	
	//check matrices dimensions
	if (nCol != mxGetN(prhs[3])) {
		mexErrMsgIdAndTxt("Trotter:", "Input 4 must have the same number of columns as input 3");
	}
	if (nRow != mxGetM(prhs[3])) {
		mexErrMsgIdAndTxt("Trotter:", "Input 4 must have the same number of rows as input 3");
	}
	if (nCol != mxGetN(prhs[4])) {
		mexErrMsgIdAndTxt("Trotter:", "Input 5 must have the same number of columns as input 3");
	}
	if (nRow != mxGetM(prhs[4])) {
		mexErrMsgIdAndTxt("Trotter:", "Input 5 must have the same number of rows as input 3");
	}
	if (nCol != mxGetN(prhs[5])) {
		mexErrMsgIdAndTxt("Trotter:", "Input 6 must have the same number of columns as input 3");
	}
	if (nRow != mxGetM(prhs[5])) {
		mexErrMsgIdAndTxt("Trotter:", "Input 6 must have the same number of rows as input 3");
	}

	double *Mext_real = mxGetPr(prhs[2]);
	double *ext_real = new double[nCol * nRow];
	double *Mext_imag = mxGetPr(prhs[3]);
	double *ext_imag = new double[nCol * nRow];
	double *Mp_real = mxGetPr(prhs[4]);
	double *p_real = new double[nCol * nRow];
	double *Mp_imag = mxGetPr(prhs[5]);
	double *p_imag = new double[nCol * nRow];

	int iterations = (int)mxGetPr(prhs[6])[0];
	int KernelType = (int)mxGetPr(prhs[7])[0];
	int *Mperiods = (int*)mxGetPr(prhs[8]);
	int *periods = new int[nCol * nRow];
	bool imag_time = (bool)mxGetPr(prhs[8])[0];

	int time[1];

	//initialize matrices
	for (int i = 0; i < nRow; i++){
		for (int j = 0; j < nCol; j++){
			ext_real[i * nCol + j] = (double)Mext_real[j * nRow + i];
			ext_imag[i * nCol + j] = (double)Mext_imag[j * nRow + i];
			p_real[i * nCol + j] = (double)Mp_real[j * nRow + i];
			p_imag[i * nCol + j] = (double)Mp_imag[j * nRow + i];
		}
	}
	for (int i = 0; i < 2; i++){
		periods[i] = (int)Mperiods[i];
	}

	//Call trotter routine
	trotter(h_a, h_b, ext_real, ext_imag, p_real, p_imag, nCol, nRow, iterations, KernelType, periods, imag_time, time);

	//Set output
	plhs[0] = mxCreateDoubleMatrix(nRow, nCol, mxREAL);
	double* Op_real = mxGetPr(plhs[0]);
	plhs[1] = mxCreateDoubleMatrix(nRow, nCol, mxREAL);
	double* Op_imag = mxGetPr(plhs[0]);
	for (int i = 0; i < nRow; i++){
		for (int j = 0; j < nCol; j++) {
			Op_real[j * nRow + i] = (double)p_real[i * nCol + j];
			Op_imag[j * nRow + i] = (double)p_imag[i * nCol + j];
		}
	}
}

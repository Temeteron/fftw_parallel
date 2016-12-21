#include "libraries.h"

typedef struct comp_comp {
	float Re;
	float Im;
} complex;

//Appropriate exponential power of omega calculation: w^ki = e^(2*PI*k*i/n)
complex omega(int n, int i, int k) {
	complex omeg;

	omeg.Re = cos(k*i*2*PI/n);
	omeg.Im = sin(k*i*2*PI/n);

	return omeg;
}

// addition of 2 complex numbers c1,c2
complex csum(complex c1, complex c2){
	complex sum;

	sum.Re = c1.Re + c2.Re;
	sum.Im = c1.Im + c2.Im;
	return sum;
}

// multiplication of 2 complex numbers c1,c2
complex cmul(complex c1, complex c2){
	complex mul;

	mul.Re = c1.Re * c2.Re - c1.Im * c2.Im;
	mul.Im = c1.Re * c2.Im + c1.Im * c2.Re;
	return mul;
}

// subtraction of 2 complex numbers c1,c2
complex csub(complex c1, complex c2){
	complex sub;

	sub.Re = c1.Re - c2.Re;
	sub.Im = c1.Im - c2.Im;
	return sub;
}

int bitrev(int inp, int numbits)
{
  int i, rev=0;
  //printf("Original int %d\n",inp);
  for (i=0; i < numbits; i++)
  {
    rev = (rev << 1) | (inp & 1);
    inp >>= 1;
  }
 // printf("Reversed int %d\n",rev);
  return rev;
}

double parallel_FFT(complex *X, complex *Y, long n,int num_thr) {
	//long ii, shift,prev_shift, l;
	long r, temp,m, i,w; //j, k, w, y, t, i2;
	//complex temp2, temp3, omeg;
	//complex t1;
	complex *R,*S,omeg;
	int start,end,element,border,tid,block,j,k;
	double en,bg,ext;
	
	R = (complex *) malloc(n*sizeof(complex));
	S = (complex *) malloc(n*sizeof(complex));

		/* Calculate r=logn with n=2^r */
	ext=0;
	bg=omp_get_wtime();
	omp_set_num_threads(num_thr);
	r=0;
	temp=n;
	while ( (n /= 2 ) != 0 ){
		r++;}
	n=temp;
	//Calculate number of iterations without communication
	
	
	for (i=0; i<n; i++){
		R[i].Re = X[i].Re;	
		R[i].Im = X[i].Im;
	}	
	
	border=n/2;	
	
	//r-d iterations with communication but since 
	
	for (m=0; m<r; m++){
		//mb=pow(2,m);
		for (i=0; i<n; i++){
			S[i].Re = R[i].Re;	
			S[i].Im = R[i].Im;
			//printf("step %ld : S has %fl \n",m,S[i].Re);
		}
		block=n/num_thr;
		
		#pragma omp parallel shared(S,border,block,m,r,n) private (element,start,end,tid,j,k,w,omeg)
		{
			tid=omp_get_thread_num();
			start=(tid)*block;
			end=start + block;
			
			
			for(element=start; element<end; element++){
				
				j=(element & (~(1 << (r-m-1)))) | (0 << (r-m-1));
				k=(element & (~(1 << (r-m-1)))) | (1 << (r-m-1));
				//Appropriate omega for each butterfly group
				w=bitrev(element,r);
				w =w << (r-1-m);
				omeg=omega(n,-1,w);
				
				if (element<k){
					//R[element].Re=S[j].Re +2*S[k].Re; 
					R[element]=csum(S[j],cmul(omeg,S[k]));
					//printf("%lf + 2* %lf \n",S[j].Re,S[k].Re);
					//printf("Thread %d at end of step %ld local R %fl %fl, j: %d k: %d \n",tid,m,R[element].Re,R[element].Im,j,k);
				}
				else {
					//R[element].Re=S[k].Re +2*S[j].Re;
					R[element]=csum(S[k],cmul(omeg,S[j]));
					//printf("%lf + 2* %lf \n",S[k].Re,S[j].Re);
					//printf("Thread %d at end of step %ld local R %fl %fl , k: %d j: %d \n",tid,m,R[element].Re,R[element].Im,k,j);
				}
			}
			{
			#pragma omp barrier
			}
			//At the end of the last step reverse indices
			if(m==r-1){
			for(element=start; element<end; element++){
				Y[element]=R[bitrev(element,r)];
				}
			
			}
		}
		border=border/2;
	}
	en=omp_get_wtime();
	ext=en-bg;
	//printf("Mean Time: %lf",en-bg);
	free(R);
	free(S);
	
	return ext;
}
	//After the process is done reverse indices in parallel
	
		
	
int main(int argc, char** argv) {	
	int size, i,num_thr;
	complex *X, *Y;
	double sum,mean,ext;
	//double start,end;
	//Provide power of size at input
    size = pow(2,atoi(argv[1]));
	num_thr = pow(2,atoi(argv[2]));
	
	if (size < num_thr || num_thr==1) {
		printf("Non optimal partitioning.Exiting\n");
		return -1;
	}
	Y = (complex *) malloc(size*sizeof(complex));
	X = (complex *) malloc(size*sizeof(complex));
	for(i=0;i<size;i++){
	
		X[i].Re=(float)rand();
		X[i].Im=(float)rand();
	}
	//for(i=0;i<size;i++) {	
	//	printf("Input: %fl %fl\n",X[i].Re,X[i].Im); }
	sum=0;
	ext=0;
	for(i=0; i<100; i++){
	//start=omp_get_wtime();
	ext=parallel_FFT(X,Y,size,num_thr);
	//end=omp_get_wtime();
	sum=sum+ext;
	}
	mean=(double) sum/ (double) 100;
	
	printf("Time spent in parallel_FFT for %d elements with %d threads: %lf\n",size,num_thr,mean);
	//printf("Time spent in serial_FFT for %d elements: %.20lf\n",size,(double)(end-start)/ CLOCKS_PER_SEC);
	//Printing the values of Y for debugging
	//for(i=0;i<size;i++) {	
		//printf("Result: %fl %fl\n",Y[i].Re,Y[i].Im); }		

	free(X);
	free(Y);
	return 0;
}
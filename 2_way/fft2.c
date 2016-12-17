#include "libraries.h"
#define N 1048576
#define N2 N/2

int main(int argc, char *argv[]) {
/* Generic version of cfft2 - no intrinsics
   W. Petersen, SAM. Math. ETHZ 1 June, 2002 */
   int first,i,icase,it,ln2,n;
   int nits=10000;
   float error,fnm1,sign,z0,z1,ggl();
   float *x,*y,*z,*w;
   static float seed;
   double t0,t1,t2,mflops,walltime(double*);
   void cffti(),cfft2();
/* allocate storage for x,y,z,w on 4-word bndr. */
   x = (float *) malloc(8*N);
   y = (float *) malloc(8*N);
   z = (float *) malloc(8*N);
   w = (float *) malloc(4*N);
   seed  = 331.0;
   n     = 2;
   for(ln2=1;ln2<21;ln2++){
      first = 1;
      for(icase=0;icase<2;icase++){
         if(first){
            for(i=0;i<2*n;i+=2){
               z0 = ggl(&seed);     /* real part of array */
               z1 = ggl(&seed);     /* imaginary part of array */
               x[i] = z0;
               z[i] = z0;           /* copy of initial real data */
               x[i+1] = z1;
               z[i+1] = z1;         /* copy of initial imag. data */
            }
         } else {
#pragma omp parallel shared(x,z,n) private(z0,z1,i)
#pragma omp for nowait
            for(i=0;i<2*n;i+=2){
               z0 = 0;              /* real part of array */
               z1 = 0;              /* imaginary part of array */
               x[i] = z0;
               z[i] = z0;           /* copy of initial real data */
               x[i+1] = z1;
               z[i+1] = z1;         /* copy of initial imag. data */
            }
         }
/* initialize sine/cosine tables */
         cffti(n,w);
/* transform forward, back */
         if(first){
            sign = 1.0;
            cfft2(n,x,y,w,sign);
            sign = -1.0;
            cfft2(n,y,x,w,sign);
/* results should be same as initial multiplied by n */
            fnm1 = 1.0/((float) n);
            error = 0.0;
            for(i=0;i<2*n;i+=2){
               error += (z[i] - fnm1*x[i])*(z[i] - fnm1*x[i]) +
                   (z[i+1] - fnm1*x[i+1])*(z[i+1] - fnm1*x[i+1]);
            }
            error = sqrt(fnm1*error);
            printf(" for n=%d, fwd/bck error=%e\n",n,error); 
            first = 0;
         } else {
            t0 = 0.0;
            t1 = walltime(&t0);
            for(it=0;it<nits;it++){
               sign = +1.0;
               cfft2(n,x,y,w,sign);
               sign = -1.0;
               cfft2(n,y,x,w,sign);
            }
            t2 = walltime(&t1);
            t1   = 0.5*t2/((double) nits);
            mflops = 5.0*((double) n)*((double) ln2)/((1.e+6)*t1);
/*          printf(" for n=%d, nits = %d\n",n,nits); 
            printf(" for n=%d, t1=%e, mflops=%e\n",n,t1,mflops); */
            printf(" %d    %e\n",n,mflops); 
         }
      }
      if((ln2%4)==0) nits /= 10;
      if(nits<1) nits = 1;
      n *= 2;
   }
   
   return 0;
}
void cffti(int n, float w[][2])
{
   int i,n2;
   float aw,arg,pi;
   pi = 3.141592653589793;
   n2 = n/2;
   aw = 2.0*pi/((float)n);
#pragma omp parallel shared(aw,w,n) private(i,arg)
#pragma omp for nowait
   for(i=0;i<n2;i++){
      arg   = aw*((float)i);
      w[i][0] = cos(arg);
      w[i][1] = sin(arg);
   }
}

// #include <math.h>
float ggl(float *ds)
{

/* generate u(0,1) distributed random numbers. 
   Seed ds must be saved between calls. ggl is 
   essentially the same as the IMSL routine RNUM. 

   W. Petersen and M. Troyer, 24 Oct. 2002, ETHZ: 
   a modification of a fortran version from 
   I. Vattulainen, Tampere Univ. of Technology, 
   Finland, 1992 */

   double t,d2=0.2147483647e10;
   t   = (float) *ds;
   t   = fmod(0.16807e5*t,d2);
   *ds = (float) t;
   return((float) ((t-1.0e0)/(d2-1.0e0)));
}
void cfft2(n,x,y,w,sign)
int n;
float x[][2],y[][2],w[][2],sign;
{
   int m, j, mj, tgle;
   void ccopy(),step();
   m    = (int) (log((float) n)/log(1.99));
   mj   = 1;
   tgle = 1;  /* toggling switch for work array */
   step(n,mj,&x[0][0],&x[n/2][0],&y[0][0],&y[mj][0],w,sign);
   if(n==2)return;
   for(j=0;j<m-2;j++){
      mj *= 2;
      if(tgle){
         step(n,mj,&y[0][0],&y[n/2][0],&x[0][0],&x[mj][0],w,sign);
         tgle = 0;
      } else {
         step(n,mj,&x[0][0],&x[n/2][0],&y[0][0],&y[mj][0],w,sign);
         tgle = 1;
      }
   }
/* last pass thru data: move y to x if needed */
   if(tgle) {
      ccopy(n,y,x);
   }
   mj   = n/2;
   step(n,mj,&x[0][0],&x[n/2][0],&y[0][0],&y[mj][0],w,sign);
}
void ccopy(int n, float x[][2], float y[][2])
{
   int i;
   for(i=0;i<n;i++){
      y[i][0] = x[i][0];
      y[i][1] = x[i][1];
   }
}
void step(n,mj,a,b,c,d,w,sign)
int n,mj;
float a[][2],b[][2],c[][2],d[][2],w[][2];
float sign;
{
   float ambr, ambu, wjw[2];
   int j, k, ja, jb, jc, jd, jw, lj, mj2;
/* one step of workspace version of CFFT2 */
   mj2 = 2*mj; lj  = n/mj2;
#pragma omp parallel shared(w,a,b,c,d,lj,mj,mj2,sign) \
 private(j,k,jw,ja,jb,jc,jd,ambr,ambu,wjw) 
/* #pragma omp for schedule(static,16) nowait */
#pragma omp for nowait
   for(j=0;j<lj;j++){
      jw = j*mj; ja  = jw; jb  = ja; jc  = j*mj2; jd  = jc;
      wjw[0] = w[jw][0]; wjw[1] = w[jw][1];
      if(sign<0.) wjw[1]=-wjw[1];
      for(k=0; k<mj; k++){
         c[jc + k][0] = a[ja + k][0] + b[jb + k][0];
         c[jc + k][1] = a[ja + k][1] + b[jb + k][1];
         ambr = a[ja + k][0] - b[jb + k][0];
         ambu = a[ja + k][1] - b[jb + k][1];
         d[jd + k][0] = wjw[0]*ambr - wjw[1]*ambu;
         d[jd + k][1] = wjw[1]*ambr + wjw[0]*ambu;
      }
   }
}
# include <sys/time.h>
double walltime(double *t0)
{
   double mic, time;
   double mega=0.000001;
   struct timeval tp;
   struct timezone tzp;
   static long base_sec = 0;
   static long base_usec = 0;

   (void) gettimeofday(&tp, &tzp);
   if (base_sec == 0) {
     base_sec  = tp.tv_sec;
     base_usec = tp.tv_usec;
   }
   time = (double)(tp.tv_sec - base_sec);
   mic = (double)(tp.tv_usec - base_usec);
   time = (time + mic * mega) - *t0;
   return(time);
}

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* Independent direct float64 reference; does not use sqlite-vec or BLAS. */
int exact_top10(const float *data, const float *queries, int n, int q, int dims,
                int cosine, int64_t *out, double *kth, double *next) {
  double *norms = NULL;
  if (cosine) {
    norms = malloc((size_t)n * sizeof(double));
    if (!norms) return -1;
    for (int i = 0; i < n; i++) {
      double sum = 0;
      for (int d = 0; d < dims; d++) {
        double x = data[(size_t)i*dims+d]; sum += x*x;
      }
      norms[i] = sqrt(sum);
      if (!(norms[i] > 0)) { free(norms); return -2; }
    }
  }
  for (int qi = 0; qi < q; qi++) {
    double best[11]; int64_t ids[11];
    for (int i = 0; i < 11; i++) { best[i] = INFINITY; ids[i] = INT64_MAX; }
    const float *query = queries + (size_t)qi*dims;
    double qnorm = 1;
    if (cosine) {
      qnorm = 0;
      for (int d = 0; d < dims; d++) { double x = query[d]; qnorm += x*x; }
      qnorm = sqrt(qnorm);
      if (!(qnorm > 0)) { free(norms); return -2; }
    }
    for (int row = 0; row < n; row++) {
      const float *v = data + (size_t)row*dims;
      double distance = 0;
      for (int d = 0; d < dims; d++) {
        double a = query[d], b = v[d];
        if (cosine) distance += a*b;
        else { double delta = a-b; distance += delta*delta; }
      }
      if (cosine) distance = 1-distance/(qnorm*norms[row]);
      if (distance < 0) distance = 0;
      if (!isfinite(distance)) { free(norms); return -2; }
      if (distance >= best[10]) continue;
      int slot = 10;
      while (slot > 0 && best[slot-1] > distance) {
        best[slot] = best[slot-1]; ids[slot] = ids[slot-1]; slot--;
      }
      best[slot] = distance; ids[slot] = row;
    }
    for (int i = 0; i < 10; i++) out[(size_t)qi*10+i] = ids[i];
    kth[qi] = best[9]; next[qi] = best[10];
  }
  free(norms); return 0;
}

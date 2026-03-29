/*
Copyright 2026 Сацук Артём Венедиктович (Satsuk Artem)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <math.h>
#include <stdint.h>

void hq_fwht_rows(float *data, int64_t rows, int32_t dim) {
    if (data == 0 || rows <= 0 || dim <= 0) {
        return;
    }

    const float scale = 1.0f / sqrtf((float)dim);
    for (int64_t row_idx = 0; row_idx < rows; ++row_idx) {
        float *row = data + (row_idx * dim);
        for (int32_t h = 1; h < dim; h <<= 1) {
            const int32_t step = h << 1;
            for (int32_t base = 0; base < dim; base += step) {
                for (int32_t j = base; j < base + h; ++j) {
                    const float a = row[j];
                    const float b = row[j + h];
                    row[j] = a + b;
                    row[j + h] = a - b;
                }
            }
        }
        for (int32_t col_idx = 0; col_idx < dim; ++col_idx) {
            row[col_idx] *= scale;
        }
    }
}

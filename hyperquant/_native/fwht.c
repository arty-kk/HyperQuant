/* SPDX-FileCopyrightText: 2026 Сацук Артём Венедиктович (Satsuk Artem) */
/* SPDX-License-Identifier: Apache-2.0 */

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

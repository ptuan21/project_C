#pragma once
// mlcpp — thư viện GPT nhỏ + lượng tử hóa, tập trung sinh văn bản tiếng Việt.
// Include file này để dùng toàn bộ. Cấu trúc theo nhóm chức năng:
//   core/   — ma trận, lưu/nạp trọng số, lượng tử hóa int8
//   prob/   — sinh số ngẫu nhiên, thống kê (kiểm thử)
//   data/   — tokenizer UTF-8
//   dl/     — autograd, layer, optimizer, lấy mẫu
//   models/ — GPT (Transformer)

// core
#include "mlcpp/core/matrix.hpp"
#include "mlcpp/core/serialize.hpp"
#include "mlcpp/core/quantize.hpp"

// prob
#include "mlcpp/prob/random.hpp"
#include "mlcpp/prob/stats.hpp"

// data
#include "mlcpp/data/tokenizer.hpp"

// dl
#include "mlcpp/dl/autograd.hpp"
#include "mlcpp/dl/nn.hpp"
#include "mlcpp/dl/sampling.hpp"

// models
#include "mlcpp/models/transformer.hpp"

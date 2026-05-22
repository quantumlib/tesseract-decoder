#ifndef BERN_UTILS_H
#define BERN_UTILS_H

namespace tesseract {

// Calculates the probability of an odd number of independent events with
// probabilities p1 and p2 occurring: p1*(1-p2) + p2*(1-p1).
double bernoulli_xor(double p1, double p2);

// Converts a probability to a log-likelihood ratio weight.
// The weight is calculated as w = ln((1-p)/p).
double to_weight(double probability);

} // namespace two_pass_decoding

#endif // BERN_UTILS_H


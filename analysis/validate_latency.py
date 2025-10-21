# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pandas as pd
import latency_model
from fit_decode_latency import parse_xlsx

if __name__ == "__main__":
    xlsx_path = "validation/full_mmlu_by_model_tegra.xlsx"
    ref_data = parse_xlsx(xlsx_path, max_entries=50, start_entry=100)

    

    data = {}
    for model_name, model_data in ref_data.items():
        data[model_name] = {}
        for key, value in model_data.items():
            data[model_name][key] = latency_model.total_latency_model(model_name, key[0], key[1])
            # print(model_name, key, value, data[model_name][key])



    # Calculate MAPE for each model
    for model_name, model_data in ref_data.items():
        print(f"\n=== MAPE Analysis for {model_name} ===")
        
        # Initialize lists to store actual and predicted values
        actual_prefill = []
        predicted_prefill = []
        actual_decode = []
        predicted_decode = []
        actual_total = []
        predicted_total = []
        
        for key, actual_values in model_data.items():
            input_tokens, output_tokens = key
            predicted_values = data[model_name][key]
            
            # Extract actual values [prefill, decode, total]
            actual_prefill.append(actual_values[0])
            actual_decode.append(actual_values[1])
            actual_total.append(actual_values[2])
            
            # Extract predicted values (prefill, decode, total)
            predicted_prefill.append(predicted_values[0])
            predicted_decode.append(predicted_values[1])
            predicted_total.append(predicted_values[2])
        
        # Calculate MAPE for each metric
        def calculate_mape(actual, predicted):
            if len(actual) == 0:
                return float('inf')
            # return (100 / len(actual)) * sum(abs((a - p) / a) for a, p in zip(actual, predicted) if a != 0)
            return (100 / len(actual)) * sum(abs((a - p) / a) for a, p in zip(actual, predicted))

        mape_prefill = calculate_mape(actual_prefill, predicted_prefill)
        mape_decode = calculate_mape(actual_decode, predicted_decode)
        mape_total = calculate_mape(actual_total, predicted_total)
        
        print(f"Prefill Latency MAPE: {mape_prefill:.2f}%")
        print(f"Decode Latency MAPE: {mape_decode:.2f}%")
        print(f"Total Latency MAPE: {mape_total:.2f}%")
        
        # # Print some sample comparisons
        # print(f"\nSample comparisons (first 5 entries):")
        # for i in range(min(5, len(actual_prefill))):
        #     print(f"  Entry {i+1}:")
        #     print(f"    Prefill: Actual={actual_prefill[i]:.3f}s, Predicted={predicted_prefill[i]:.3f}s")
        #     print(f"    Decode: Actual={actual_decode[i]:.3f}s, Predicted={predicted_decode[i]:.3f}s")
        #     print(f"    Total: Actual={actual_total[i]:.3f}s, Predicted={predicted_total[i]:.3f}s")
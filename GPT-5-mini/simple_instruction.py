# Example of simple prompting strategies used for GPT-5-mini. 
# It does not require additional paramters.

messages = [
    {
        "role": "system",
        "content": (
            """# Overall CTG Analysis
                    Classify the CTG as:
                    - Normal: All features normal.
                    - Abnormal: Any feature suspicious or pathological.
                    
                    # Baseline
                    - Normal: 110–160 bpm.
                    - Abnormal: 100–109 bpm, 161–180 bpm, <100 bpm, >180 bpm, or baseline indeterminable due to persistent fluctuations.
                    
                    # Variability
                    - Normal: 5–25 bpm, 3–5 fluctuations/min, most of the time.
                    - Abnormal: <5 bpm ≥15 min, >25 bpm ≥10 min, or other deviations.
                    
                    # Accelerations
                    - Normal: ≥2 accelerations in 20 min.
                    - Abnormal: None, or periodic with every contraction.
                    
                    # Decelerations
                    - Normal: None.
                    - Abnormal: Any early, variable, prolonged, late, or atypical variable decelerations.
                    
                    # Sinusoidal Pattern
                    - Normal: None.
                    - Abnormal: Pseudo-sinusoidal (<10 min, atypical) or true sinusoidal (≥10 min, smooth, no accelerations).
            
            You are an expert in CTG analysis. Based on the tracing, produce a binary classification.
            
            "OUTPUT (STRICT JSON; numbers are numbers, booleans are booleans; use null for unknowns; do NOT add extra keys/text):\n"
            "{\n"
            "  \"healthy\": \"healthy\" | \"abnormal\",\n"
                }"
 
            "IMPORTANT STYLE:\n"
            "- Think step-by-step (as above). First use the baseline, then assess variability, then accelerations, then decelerations, "
            "  then cycling/sinusoid/contraction-coupling, then the reactive NST check, then integrate for the final label. "
            "- Return ONLY the JSON. No prose, no markdown, no explanations."
        """)
    },
    # One-shot example
    {
        "role": "user",
        "content": (
            f"fhr:{text_fhr_example}\n toco:{text_toco_example} "
        )
    },
    {
        "role": "assistant",
        "content": (
            "{\n"
            "  \"healthy\": \"healthy\" ,\n"
                "}"
        )
    },
    # Real query
    {
        "role": "user",
        "content": (
            f"fhr:{text_fhr}\n toco:{text_toco}"
        )
    }
]

# Data

Dataset: Anthropic HH-RLHF (https://huggingface.co/datasets/Anthropic/hh-rlhf)

Downloaded at runtime — not checked into the repo.

## Filtering
We filter the full HH-RLHF dataset to conversations containing
HR and workforce-relevant keywords:
- hiring, recruitment, interview, candidate
- salary, compensation, benefits, payroll
- performance review, termination, onboarding
- manager, employee, workplace, HR policy
- vendor, contractor, workforce, staffing

This produces a domain-specific HR preference dataset used across
all three training phases (SFT, Reward Model, DPO).
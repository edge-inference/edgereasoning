#!/usr/bin/env python3
"""
Create realistic synthetic questions with exact token counts for decode energy studies.
Supports both Qwen and Llama tokenizers with diverse, coherent content.
"""

import csv
import random
import re
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import argparse

# Realistic question templates based on natural research/analysis prompts
REALISTIC_QUESTION_STARTERS = [
    "Review the available evidence on {topic} interventions. How convincing is the empirical support, and what are the key limitations?",
    "Analyze the effectiveness of {approach} in {context}. What factors contribute to successful implementation?",
    "Evaluate the current state of {field} research. What gaps exist in our understanding, and how might they be addressed?",
    "Examine the relationship between {factor_a} and {factor_b} in {setting}. What mechanisms explain this connection?",
    "Assess the impact of {policy_change} on {outcome}. Consider both intended and unintended consequences.",
    "Compare different approaches to {challenge}. Which strategies show the most promise and why?",
    "Investigate how {system} adapts to {change}. What are the implications for {stakeholder_group}?",
    "Discuss the role of {key_factor} in {process}. How does this influence {result}?",
    "Analyze recent developments in {domain}. What trends are emerging and what do they suggest for the future?",
    "Evaluate the trade-offs between {option_a} and {option_b} when addressing {problem}. What factors should guide decision-making?",
]

# Realistic content pools for natural question generation
REALISTIC_CONTENT = {
    "topic": ["community-level health", "educational technology", "environmental policy", "workplace diversity", 
              "digital transformation", "mental health", "urban planning", "renewable energy adoption", "social media regulation", "healthcare access"],
    "approach": ["evidence-based interventions", "participatory methods", "technology-driven solutions", "policy reforms", 
                "community engagement strategies", "data-driven approaches", "collaborative partnerships", "preventive measures"],
    "context": ["low-resource settings", "urban environments", "rural communities", "developing countries", 
               "post-pandemic recovery", "organizational change", "policy implementation", "community development"],
    "field": ["public health", "education policy", "environmental science", "organizational psychology", 
             "social work", "urban studies", "technology adoption", "healthcare delivery"],
    "factor_a": ["socioeconomic status", "organizational culture", "policy implementation", "community engagement", 
                "resource availability", "technological readiness", "stakeholder buy-in", "regulatory frameworks"],
    "factor_b": ["health outcomes", "educational achievement", "environmental sustainability", "economic development", 
                "social cohesion", "innovation adoption", "service quality", "community resilience"],
    "setting": ["healthcare systems", "educational institutions", "workplace environments", "community organizations", 
               "government agencies", "non-profit sectors", "international development", "technology companies"],
    # Additional placeholders for base question templates
    "policy_change": ["new regulations", "funding reforms", "governance restructuring", "service delivery changes", 
                     "quality standards", "accountability measures", "capacity building initiatives", "technology adoption"],
    "outcome": ["service quality", "community health", "educational achievement", "economic development", 
               "environmental sustainability", "social cohesion", "innovation adoption", "organizational effectiveness"],
    "challenge": ["resource constraints", "implementation barriers", "coordination difficulties", "sustainability issues", 
                 "quality improvement", "stakeholder engagement", "capacity building", "system integration"],
    "system": ["healthcare delivery", "educational provision", "social services", "governance structures", 
              "technology platforms", "community networks", "organizational systems", "policy frameworks"],
    "change": ["policy reforms", "technological advancement", "demographic shifts", "resource changes", 
              "regulatory updates", "organizational restructuring", "community needs evolution", "external pressures"],
    "stakeholder_group": ["communities", "service users", "practitioners", "policy makers", 
                         "organizations", "funding bodies", "oversight agencies", "advocacy groups"],
    "key_factor": ["leadership", "resource availability", "stakeholder engagement", "organizational capacity", 
                  "policy support", "technical expertise", "community buy-in", "implementation quality"],
    "process": ["service delivery", "policy implementation", "capacity building", "quality improvement", 
               "stakeholder engagement", "system development", "performance monitoring", "innovation adoption"],
    "result": ["improved outcomes", "enhanced capacity", "better coordination", "increased effectiveness", 
              "stronger sustainability", "greater equity", "improved quality", "enhanced innovation"],
    "domain": ["healthcare", "education", "social services", "environmental policy", 
              "technology adoption", "community development", "organizational management", "public administration"],
    "option_a": ["centralized approaches", "technology-driven solutions", "community-based models", "evidence-based methods"],
    "option_b": ["decentralized strategies", "relationship-focused approaches", "institution-led initiatives", "practice-based methods"],
    "problem": ["service gaps", "coordination challenges", "resource limitations", "quality concerns", 
              "sustainability issues", "equity disparities", "capacity constraints", "implementation barriers"],
    # Placeholders for narrative structure
    "historical_focus": ["top-down policy mandates", "isolated pilot programs", "technology-centric solutions", "academic research"],
    "new_focus": ["community-led initiatives", "integrated systems approaches", "participatory design", "implementation science"],
    "key_finding": ["contextual factors were the primary driver of success", "stakeholder buy-in was essential for sustainability", "early-stage adaptation was critical", "the intervention had unintended negative consequences"],
    "policy_factor_1": ["recent legislative changes", "shifting funding priorities", "new international agreements", "public opinion trends"],
    "policy_factor_2": ["inter-agency dynamics", "the influence of advocacy networks", "economic conditions", "technological disruption"],
    # Placeholders for new narrative structures
    "objective": ["assess the scalability of the intervention", "identify key drivers of success", "develop a framework for future implementation", "evaluate the cost-effectiveness of the approach"],
    "method": ["a comparative case study", "a longitudinal analysis", "a mixed-methods evaluation", "a theory-based review"],
    "challenge_area": ["ensuring stakeholder buy-in", "securing long-term funding", "adapting to local contexts", "measuring long-term impact"],
    "implication": ["for future policy design", "for practitioner training", "for resource allocation", "for community engagement strategies"],
    "scenario": ["a public health crisis in an urban area", "a large-scale technology adoption in education", "an environmental policy change in a rural community", "a workplace diversity initiative in a multinational corporation"],
    "constraint": ["limited financial resources", "a tight implementation timeline", "political opposition", "a lack of technical expertise"],
    "intervention_option": ["a community-based program", "a technology-driven solution", "a policy reform initiative", "a capacity-building program"],
    "evaluation_criterion": ["cost-effectiveness", "stakeholder satisfaction", "sustainability", "equity of outcomes"],
    "literature_topic": ["implementation science", "organizational change", "community development", "public health interventions"],
    "gap_in_literature": ["a lack of long-term outcome data", "an over-reliance on self-reported measures", "a failure to account for contextual factors", "limited understanding of causal mechanisms"],
    "research_question": ["how can implementation fidelity be improved", "what are the unintended consequences of the intervention", "how can the program be adapted for different contexts", "what factors predict long-term sustainability"],
    "expected_outcome": ["a clearer understanding of best practices", "an evidence-based framework for decision-making", "actionable recommendations for policymakers", "a validated model for program delivery"],
    "factor_1": ["regulatory frameworks", "technological capabilities", "cultural norms", "economic incentives", 
                 "institutional capacity", "resource availability", "stakeholder engagement", "market dynamics", "political stability", "social capital"],
    "factor_2": ["implementation strategies", "funding mechanisms", "public awareness", "technical expertise", 
                 "infrastructure quality", "policy coordination", "community participation", "international cooperation", "risk management", "performance monitoring"],
    "system": ["healthcare delivery", "educational outcomes", "financial markets", "transportation networks", 
               "energy systems", "food security", "urban planning", "environmental management", "innovation ecosystems", "social services"],
    "variable": ["demographic shifts", "technological disruption", "policy changes", "economic fluctuations", 
                "climate variations", "social movements", "regulatory updates", "market competition", "resource scarcity", "cultural evolution"],
    "outcome": ["improved efficiency", "enhanced equity", "reduced emissions", "increased accessibility", 
               "better health outcomes", "stronger resilience", "greater innovation", "improved sustainability", "enhanced security", "increased prosperity"],
    "intervention": ["digital transformation", "policy reform", "infrastructure investment", "capacity building", 
                    "regulatory change", "technology adoption", "community engagement", "international cooperation", "research initiatives", "public-private partnerships"],
    "target_population": ["small businesses", "rural communities", "urban youth", "elderly populations", 
                         "healthcare workers", "students", "entrepreneurs", "low-income families", "marginalized groups", "public sector employees"],
    "approach_a": ["centralized coordination", "market-based solutions", "technology-driven approaches", "community-led initiatives", "regulatory enforcement"],
    "approach_b": ["decentralized networks", "government intervention", "traditional methods", "top-down management", "collaborative partnerships"],
    "problem": ["resource allocation", "service delivery", "quality assurance", "risk management", "stakeholder coordination"],
    "scenario": ["resource-constrained environments", "rapidly changing conditions", "high-uncertainty contexts", "multi-stakeholder situations", "time-sensitive circumstances"],
    "method_1": ["quantitative analysis", "participatory approaches", "evidence-based practices", "adaptive management", "systems thinking"],
    "method_2": ["qualitative assessment", "expert consultation", "best practice adoption", "standardized procedures", "linear planning"],
    "domain": ["public health", "environmental management", "economic development", "social services", "technology implementation"],
    "option_a": ["incremental improvements", "comprehensive reform", "targeted interventions", "preventive measures", "capacity enhancement"],
    "option_b": ["transformational change", "maintaining status quo", "broad-based programs", "reactive responses", "resource reallocation"],
    "situation": ["crisis response", "long-term planning", "performance optimization", "risk mitigation", "innovation adoption"],
    "challenge": ["coordination failures", "resource constraints", "information asymmetries", "stakeholder conflicts", "technical limitations"],
    "constraint_1": ["limited funding", "regulatory restrictions", "technical capabilities", "time pressures", "political considerations"],
    "constraint_2": ["stakeholder requirements", "environmental standards", "quality expectations", "scalability needs", "sustainability goals"],
    "objective": ["cost-effectiveness", "social impact", "environmental sustainability", "stakeholder satisfaction", "long-term viability"],
    "issue": ["service gaps", "performance deficits", "coordination challenges", "resource inefficiencies", "quality variations"],
    "requirement_1": ["transparency", "accountability", "efficiency", "equity", "sustainability"],
    "requirement_2": ["stakeholder engagement", "evidence-based decisions", "continuous improvement", "risk management", "innovation"],
    "complex_system": ["multi-level governance", "integrated service delivery", "cross-sector collaboration", "adaptive networks", "dynamic partnerships"],
    "uncertainty_type": ["technological change", "policy shifts", "market volatility", "environmental variability", "social evolution"],
    "process": ["decision-making", "resource allocation", "service delivery", "quality control", "performance monitoring"],
    "environment": ["competitive markets", "regulated industries", "public sector", "non-profit organizations", "international contexts"],
    "component_1": ["governance structures", "operational processes", "stakeholder networks", "information systems", "resource flows"],
    "component_2": ["feedback mechanisms", "quality assurance", "performance metrics", "risk controls", "innovation processes"],
    "cause": ["policy changes", "technological advances", "demographic shifts", "economic conditions", "environmental factors"],
    "effect": ["behavioral changes", "performance outcomes", "system adaptations", "resource reallocations", "structural reforms"],
    "system_context": ["organizational settings", "community environments", "market conditions", "institutional frameworks", "cultural contexts"],
    "independent_var": ["investment levels", "policy interventions", "technological adoption", "training programs", "organizational changes"],
    "dependent_var": ["performance indicators", "outcome measures", "satisfaction levels", "efficiency metrics", "impact assessments"],
    "conditions": ["different sectors", "various contexts", "multiple timeframes", "diverse populations", "varying scales"],
    "policy": ["healthcare reform", "education policy", "environmental regulation", "economic stimulus", "social programs"],
    "goal": ["improved outcomes", "increased efficiency", "enhanced equity", "reduced costs", "better quality"],
    "practice": ["service delivery", "resource management", "stakeholder engagement", "performance monitoring", "quality assurance"],
    "field": ["public administration", "healthcare management", "environmental science", "social work", "economic development"],
    "hypothesis": ["causal relationships", "intervention effectiveness", "system performance", "behavioral patterns", "outcome predictors"],
    "data_source": ["longitudinal studies", "cross-sectional surveys", "administrative records", "experimental data", "observational research"],
}

# Pools for expansion content
natural_pools = {
    "stakeholder_type": ["community leaders", "advocacy groups", "local stakeholders", "implementing organizations", "policy makers"],
    "implementation_factor": ["resource availability", "cultural alignment", "policy constraints", "organizational capacity", "technical readiness"],
    "outcome_type": ["priorities and outcomes", "implementation success", "community engagement", "service delivery", "long-term sustainability"],
    "actor_group": ["researchers", "practitioners", "organizations", "communities", "institutions"],
    "challenge_type": ["contextual factors", "implementation barriers", "resource constraints", "coordination challenges", "sustainability issues"],
    "specific_challenge_1": ["resource availability", "staff capacity", "community buy-in", "regulatory compliance", "funding stability"],
    "specific_challenge_2": ["cultural alignment", "technical expertise", "stakeholder coordination", "policy support", "organizational readiness"],
    "specific_challenge_3": ["measurement systems", "quality assurance", "scalability planning", "risk management", "continuous improvement"],
    "measure_type": ["cost-effectiveness", "implementation fidelity", "outcome measurement", "stakeholder satisfaction", "sustainability indicators"],
    "constraint_type": ["regulatory requirements", "resource limitations", "time pressures", "technical constraints", "political considerations"],
    "process_type": ["implementation", "service delivery", "quality improvement", "stakeholder engagement", "system development"],
    "system_component": ["organizational processes", "governance structures", "quality systems", "coordination mechanisms", "feedback loops"],
    "component_1": ["continuous improvement", "performance monitoring", "stakeholder engagement", "resource allocation", "risk management"],
    "component_2": ["coordination across sectors", "capacity building", "innovation adoption", "quality assurance", "knowledge management"],
    "component_3": ["partnership development", "community participation", "evidence integration", "system adaptation", "outcome tracking"],
    "goal_type": ["measurable impacts", "sustainable outcomes", "system improvements", "stakeholder value", "community benefit"],
    "study_type": ["studies", "evaluations", "assessments", "reviews", "analyses"],
    "method_type": ["theory-driven evaluation methods", "participatory approaches", "mixed-methods designs", "comparative analyses", "longitudinal studies"],
    "challenge_category": ["sustainability challenges", "implementation barriers", "coordination difficulties", "resource constraints", "capacity limitations"],
    "capacity_type": ["technical capacity building", "organizational development", "staff training", "system strengthening", "infrastructure improvement"],
    "contextual_factor": ["organizational culture", "community context", "policy environment", "stakeholder relationships", "resource ecosystem"],
    "factor_group": ["stakeholder engagement, policy alignment, and system readiness", "resource adequacy, technical capacity, and organizational support", "community buy-in, leadership commitment, and implementation quality"],
    "scalability_aspect": ["scalability", "sustainability", "adaptability", "transferability", "replicability"],
    "timeframe_1": ["short-term outcomes", "immediate impacts", "initial results", "early indicators", "preliminary findings"],
    "timeframe_2": ["long-term effectiveness", "sustained impact", "lasting change", "system transformation", "enduring benefits"],
    "sector": ["the field", "policy development", "practice improvement", "system reform", "community development"],
    "related_context": ["similar settings", "comparable contexts", "related domains", "other sectors", "different populations"],
    "barrier_1": ["resource constraints", "stakeholder resistance", "technical limitations", "regulatory barriers", "organizational inertia"],
    "barrier_2": ["coordination challenges", "capacity gaps", "competing priorities", "political pressures", "sustainability concerns"],
}

# Expansion templates for building questions
natural_expansions = [
    "In your analysis, consider the influence of {stakeholder_type} and {implementation_factor}, especially in shaping {outcome_type}.",
    "Discuss how {actor_group} have addressed {challenge_type} such as {specific_challenge_1}, {specific_challenge_2}, and {specific_challenge_3}.",
    "Evaluate whether evidence on {measure_type} is well documented and how {constraint_type} might affect {process_type}.",
    "Explore the role of {system_component} like {component_1}, {component_2}, and {component_3} in achieving {goal_type}.",
    "Compare findings from different {study_type}, noting where {method_type} have added clarity or revealed new insights.",
    "Address {challenge_category}, {capacity_type}, and the sometimes overlooked importance of {contextual_factor}.",
]

# Paraphrased alternatives for common "reflective" expansions to reduce echo
reflective_expansions = {
    "implications": [
        "Consider the broader implications for {sector} and how lessons learned might apply to {related_context}.",
        "What are the wider consequences for {sector}, and how could these findings be relevant in {related_context}?",
        "Think about what this means for {sector} as a whole, and its applicability to {related_context}.",
        "How might these results influence {sector}, and what is their relevance for {related_context}?",
        "What are the potential impacts on {sector}, and how could this knowledge be transferred to {related_context}?",
    ],
    "examples": [
        "When presenting your answer, provide specific examples where possible, and explain how they relate to both {timeframe_1} and {timeframe_2}.",
        "Use concrete examples to illustrate your points, connecting them to both {timeframe_1} and {timeframe_2}.",
        "Support your analysis with specific cases, and discuss their connection to {timeframe_1} and {timeframe_2}.",
        "Illustrate your arguments with real-world examples, linking them to {timeframe_1} and {timeframe_2}.",
        "Provide clear examples to back up your claims, and explain their relevance to {timeframe_1} and {timeframe_2}.",
    ],
    "barriers": [
        "Examine potential barriers including {barrier_1}, {barrier_2}, and strategies for overcoming these challenges.",
        "Analyze the key obstacles, such as {barrier_1} and {barrier_2}, and propose ways to address them.",
        "What are the main challenges, including {barrier_1} and {barrier_2}, and how can they be surmounted?",
        "Investigate the significant hurdles, like {barrier_1} and {barrier_2}, and suggest methods for tackling them.",
        "Consider the major impediments, for instance {barrier_1} and {barrier_2}, and outline strategies to mitigate them.",
    ],
}


def load_tokenizer(model_name: str):
    """Load tokenizer with error handling."""
    try:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return None

def fill_template(template: str, content_pools: Dict[str, List[str]], ignore: List[str] = []) -> str:
    """Fill a template with random content from pools."""
    filled = template
    placeholders = re.findall(r'\{(\w+)\}', template)
    
    for placeholder in placeholders:
        if placeholder in ignore:
            continue
        if placeholder in content_pools:
            replacement = random.choice(content_pools[placeholder])
            filled = filled.replace(f'{{{placeholder}}}', replacement)
        else:
            # Fallback for missing placeholders
            filled = filled.replace(f'{{{placeholder}}}', f"[{placeholder}]")
    
    return filled

def generate_realistic_base_question() -> str:
    """Generate a realistic base question using natural templates."""
    starter = random.choice(REALISTIC_QUESTION_STARTERS)
    return fill_template(starter, REALISTIC_CONTENT)

def _remove_consecutive_duplicate_sentences(text: str) -> str:
    """Remove immediate duplicate sentences to reduce echo without altering content order."""
    # Simple sentence split on period/question/exclamation
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned_parts: List[str] = []
    previous = None
    for p in parts:
        if not p:
            continue
        if previous is not None and p == previous:
            continue
        cleaned_parts.append(p)
        previous = p
    return ' '.join(cleaned_parts)


def _extend_with_content(text: str) -> str:
    """Append one substantive content expansion to the text."""
    # Prefer content expansions (non-reflective)
    content_templates = [e for e in natural_expansions if "{sector}" not in e and "{timeframe_1}" not in e and "{barrier_1}" not in e]
    template = random.choice(content_templates) if content_templates else random.choice(natural_expansions)
    return text + " " + fill_template(template, natural_pools)


def add_natural_expansion(question: str, target_tokens: int, current_tokens: int, tokenizer) -> str:
    """Add natural contextual information to reach target token count."""
    
    expanded = question
    
    # Separate content-adding from reflective expansions
    content_expansions = [e for e in natural_expansions if "{sector}" not in e and "{timeframe_1}" not in e and "{barrier_1}" not in e]
    
    # Get one of each type of reflective expansion, with paraphrasing
    implication_expansion = random.choice(reflective_expansions["implications"])
    example_expansion = random.choice(reflective_expansions["examples"])
    barrier_expansion = random.choice(reflective_expansions["barriers"])
    
    # We'll use reflective additions at most once near the end
    used_reflectives = False
    
    # Prepare the first pipeline of content expansions
    random.shuffle(content_expansions)
    expansion_pipeline: List[str] = list(content_expansions)
    expansion_idx = 0
    
    while True:
        # Tokenize current text
        current_token_count = len(tokenizer.encode(expanded, add_special_tokens=False))
        
        if current_token_count >= target_tokens - 5:  # Allow small tolerance
            break
            
        # If we've exhausted the current pipeline, decide what to append next
        if expansion_idx >= len(expansion_pipeline):
            # If we haven't added reflectives yet and we're close to target, add them once
            if not used_reflectives and current_token_count >= max(64, int(0.6 * target_tokens)):
                expansion_pipeline = [barrier_expansion, implication_expansion, example_expansion]
                expansion_idx = 0
                used_reflectives = True
            else:
                # Refill with new shuffled content expansions to keep adding substance
                random.shuffle(content_expansions)
                expansion_pipeline = list(content_expansions)
                expansion_idx = 0
        
        # Add next expansion from the pipeline
        template = expansion_pipeline[expansion_idx]
        expansion_idx += 1
        
        expansion = fill_template(template, natural_pools)
        expanded += " " + expansion
        
        # Safety check to avoid infinite loop
        if len(expanded) > target_tokens * 15:  # Rough character-to-token ratio safety
            break
    
    return expanded


def trim_to_target_tokens(text: str, target_tokens: int, tokenizer) -> str:
    """Trim text to exact target token count with padding if needed."""
    # Clean simple echoes before computing tokens
    text = _remove_consecutive_duplicate_sentences(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Grow content first if too short
    grow_attempts = 0
    while len(tokens) < target_tokens - 8 and grow_attempts < 8:
        text = _extend_with_content(text)
        text = _remove_consecutive_duplicate_sentences(text)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        grow_attempts += 1
    
    if len(tokens) <= target_tokens:
        # Need to pad - add concise, varied filler content with a hard cap
        padding_phrases = [
            "Provide a concise example.",
            "Briefly justify your reasoning.",
            "Reference one comparable case.",
            "Note any key assumptions.",
            "Clarify scope and limitations.",
            "Summarize the expected impact.",
            "Highlight trade-offs briefly.",
            "State relevant stakeholders.",
            "Mention data sources succinctly.",
            "Outline evaluation criteria.",
        ]
        random.shuffle(padding_phrases)
        pad_idx = 0
        max_pad = 6
        
        padded_text = text
        while len(tokenizer.encode(padded_text, add_special_tokens=False)) < target_tokens and pad_idx < max_pad and pad_idx < len(padding_phrases):
            padding = padding_phrases[pad_idx]
            pad_idx += 1
            padded_text += " " + padding
        
        # If still short after limited padding, grow content again
        grow_attempts_2 = 0
        while len(tokenizer.encode(padded_text, add_special_tokens=False)) < target_tokens - 2 and grow_attempts_2 < 4:
            padded_text = _extend_with_content(padded_text)
            padded_text = _remove_consecutive_duplicate_sentences(padded_text)
            grow_attempts_2 += 1
        
        tokens = tokenizer.encode(padded_text, add_special_tokens=False)
        text = padded_text
    
    # Now trim to exact target
    if len(tokens) > target_tokens:
        trimmed_tokens = tokens[:target_tokens]
        trimmed_text = tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
        return _remove_consecutive_duplicate_sentences(trimmed_text)
    
    # If we're within 2 tokens, that's close enough
    final_tokens = len(tokens)
    if abs(final_tokens - target_tokens) <= 2:
        return text
    
    # Fallback exact trim
    trimmed_tokens = tokens[:target_tokens]
    return tokenizer.decode(trimmed_tokens, skip_special_tokens=True)


def generate_question_for_target_tokens(target_tokens: int, tokenizer) -> str:
    """Generate a realistic, coherent question with exact target token count."""
    
    # Generate realistic base question
    base_question = generate_realistic_base_question()
    base_tokens = len(tokenizer.encode(base_question, add_special_tokens=False))
    
    # Keep expanding content until close to target, then trim
    expanded = base_question
    while base_tokens < target_tokens - 8:
        expanded = add_natural_expansion(expanded, target_tokens, base_tokens, tokenizer)
        expanded = _remove_consecutive_duplicate_sentences(expanded)
        base_tokens = len(tokenizer.encode(expanded, add_special_tokens=False))
        if len(expanded) > target_tokens * 15:
            break
    
    final_question = trim_to_target_tokens(expanded, target_tokens, tokenizer)
    return final_question

def generate_narrative_question(target_tokens: int, tokenizer) -> str:
    """Generate a long-form, narrative-style question for a more realistic feel."""

    narrative_templates = [
        """
Introduction:
The focus on {topic} has shifted from {historical_focus} to {new_focus}. Early research showed that {key_finding}, but the landscape is evolving due to {policy_factor_1} and {policy_factor_2}. This analysis is intended to synthesize current understanding and identify key gaps.

Problem Statement:
Despite progress in {domain}, significant challenges remain in addressing {problem}. A comprehensive review is needed to evaluate the effectiveness of current strategies and inform future interventions.

Core Inquiry:
Your task is to conduct a detailed analysis addressing the following points. Please ensure your response is well-supported with evidence and examples.
{inquiry_points}
""",
        """
Background:
Recent developments in {domain} have highlighted the need to re-evaluate current approaches to {problem}. This analysis will build on existing work by focusing on {topic}.

Objectives:
The primary objective is to {objective}. This will be achieved by examining the issue through the lens of {new_focus}.

Methods:
The proposed method is {method}, which will allow for a detailed exploration of the key variables.

Challenges:
A key challenge in this area is {challenge_area}. This analysis will pay close attention to this issue.

Policy Implications:
The findings of this analysis will have significant implications {implication}.

Core Inquiry:
Please provide a detailed analysis of the following points, using specific examples to support your arguments.
{inquiry_points}
""",
        """
Scenario:
Consider the following scenario: {scenario}. This situation is complicated by {constraint}.

Intervention Options:
Several intervention options are being considered, including {intervention_option}.

Evaluation Criteria:
The success of any intervention will be judged based on {evaluation_criterion}.

Core Inquiry:
Please evaluate the potential intervention options based on the criteria provided. Your analysis should address the following points in detail.
{inquiry_points}
""",
        """
Literature Review:
A review of the literature on {literature_topic} reveals {gap_in_literature}.

Gap Analysis:
This gap in our understanding is significant because it limits our ability to address {problem}.

Research Questions:
This analysis will address the following research question: {research_question}.

Expected Outcomes:
The expected outcome of this analysis is {expected_outcome}.

Core Inquiry:
To address the research question, please provide a thorough analysis of the following points.
{inquiry_points}
"""
    ]
    
    # Choose a random narrative template
    narrative_template = random.choice(narrative_templates)
    
    # Fill the main narrative template, but ignore the inquiry_points placeholder for now
    base_narrative = fill_template(narrative_template, REALISTIC_CONTENT, ignore=['inquiry_points'])
    
    # Dynamically build the inquiry section to reach the target token count
    inquiry_section = ""

    # Build the weighted expansion pipeline to ensure variety and natural flow
    content_expansions = [e for e in natural_expansions if "{sector}" not in e and "{timeframe_1}" not in e and "{barrier_1}" not in e]
    implication_expansion = random.choice(reflective_expansions["implications"])
    example_expansion = random.choice(reflective_expansions["examples"])
    barrier_expansion = random.choice(reflective_expansions["barriers"])
    random.shuffle(content_expansions)
    content_idx = 0
    used_reflectives = False

    while True:
        current_tokens = len(tokenizer.encode(base_narrative.replace('{inquiry_points}', inquiry_section), add_special_tokens=False))
        if current_tokens >= target_tokens - 8:  # Closer margin for accuracy
            break

        # Add reflective set once when close to target
        if (not used_reflectives) and current_tokens >= max(64, int(0.6 * target_tokens)):
            for tmpl in [barrier_expansion, implication_expansion, example_expansion]:
                expansion = fill_template(tmpl, natural_pools)
                inquiry_section += f"- {expansion}\n"
            used_reflectives = True
            continue

        # Refill content templates if exhausted
        if content_idx >= len(content_expansions):
            random.shuffle(content_expansions)
            content_idx = 0

        # Append one content item
        template = content_expansions[content_idx]
        content_idx += 1
        expansion = fill_template(template, natural_pools)
        inquiry_section += f"- {expansion}\n"

        # Safety break
        if len(inquiry_section) > target_tokens * 15:
            break

    final_narrative = base_narrative.replace('{inquiry_points}', inquiry_section)
    return trim_to_target_tokens(final_narrative, target_tokens, tokenizer)


def create_realistic_synthetic_dataset(
    model_name: str, 
    target_tokens: List[int], 
    output_file: str,
    questions_per_length: int = 1
):
    """Create synthetic dataset with realistic questions."""
    
    print(f"Creating realistic synthetic dataset for {model_name}")
    print(f"Target tokens: {target_tokens}")
    print(f"Output file: {output_file}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    if tokenizer is None:
        print(f"Failed to load tokenizer for {model_name}")
        return
    
    questions = []
    question_id = 1000
    
    for token_count in target_tokens:
        print(f"Generating questions for {token_count} tokens...")
        
        for i in range(questions_per_length):
            try:
                # Generate question using the appropriate method
                question_text = generate_narrative_question(token_count, tokenizer)
                
                # Verify token count
                actual_tokens = len(tokenizer.encode(question_text, add_special_tokens=False))
                
                question_data = {
                    'input_tokens': actual_tokens,
                    'question': question_text,
                    'choices': "['A', 'B', 'C', 'D']",
                    'output_tokens': 1,
                    'question_id': question_id,
                    'subject': 'synthetic_analysis',
                    'correct_answer': 'A' 
                }
                
                questions.append(question_data)
                question_id += 10
                
                print(f"  Generated question {question_id-10}: {actual_tokens} tokens (target: {token_count})")
                
            except Exception as e:
                print(f"Error generating question for {token_count} tokens: {e}")
                continue
    
    # Write to CSV
    print(f"Writing {len(questions)} questions to {output_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['input_tokens', 'question', 'choices', 'output_tokens', 'question_id', 'subject', 'correct_answer']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions)
    
    print(f"Successfully created {output_file}")
    
    # Print statistics
    actual_tokens = [q['input_tokens'] for q in questions]
    print(f"\nToken count statistics:")
    print(f"  Min: {min(actual_tokens)}")
    print(f"  Max: {max(actual_tokens)}")
    print(f"  Average: {sum(actual_tokens) / len(actual_tokens):.1f}")

def main():
    parser = argparse.ArgumentParser(description='Create realistic synthetic questions for decode energy studies')
    parser.add_argument('--models', nargs='+', default=['meta-llama/Llama-2-7b-hf', 'Qwen/Qwen2.5-7B'], 
                       help='Model names for tokenizers')
    parser.add_argument('--token-range', nargs=2, type=int, default=[128, 4096], 
                       help='Token range (min, max)')
    parser.add_argument('--token-step', type=int, default=128, 
                       help='Token increment step')
    parser.add_argument('--questions-per-length', type=int, default=1, 
                       help='Number of questions per token length')
    
    args = parser.parse_args()
    
    # Generate target token counts
    target_tokens = list(range(args.token_range[0], args.token_range[1] + args.token_step, args.token_step))
    
    for model_name in args.models:
        # Use simple naming based on tokenizer type
        if 'qwen' in model_name.lower():
            output_file = "datasets/synthetic_qwen.csv"
        elif 'llama' in model_name.lower():
            output_file = "datasets/synthetic_llama.csv"
        else:
            # Fallback to original naming
            clean_name = model_name.replace('/', '_').replace('-', '_').lower()
            output_file = f"datasets/synthetic_{clean_name}.csv"
        
        create_realistic_synthetic_dataset(
            model_name=model_name,
            target_tokens=target_tokens,
            output_file=output_file,
            questions_per_length=args.questions_per_length
        )

if __name__ == "__main__":
    main()

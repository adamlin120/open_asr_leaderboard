from enum import Enum


class InputType(Enum):
    NORMALIZED = "Normalized"
    PUNCTUATED = "Punctuated"
    PUNCTUATED_CASED = "Punctuated & Cased"


class OutputType(Enum):
    NORMALIZED = "Normalized"
    PUNCTUATED = "Punctuated"
    PUNCTUATED_CASED = "Punctuated & Cased"


def generate_instructions(input_type, output_type):
    """
    Generate instructions for an ASR correction task based on the input and output types.

    Args:
        input_type (InputType): The type of input hypotheses.
        output_type (OutputType): The desired type of output transcription.

    Returns:
        str: The generated instructions for the ASR correction task.

    Raises:
        ValueError: If an invalid input or output type is provided.

    Example:
        >>> input_type = InputType.PUNCTUATED_CASED
        >>> output_type = OutputType.PUNCTUATED
        >>> instructions = generate_instructions(input_type, output_type)
        >>> print(instructions)
        The following text contains 5-best hypotheses from an Automatic Speech Recognition system. As part of a speech recognition task, please perform error correction on the hypotheses to generate the most accurate transcription of the spoken text. The input hypotheses include both punctuation and proper case information. Your output transcription should include punctuation but not be case-sensitive.
    """
    instructions = "The following text contains 5-best hypotheses from an Automatic Speech Recognition system. As part of a speech recognition task, please perform error correction on the hypotheses to generate the most accurate transcription of the spoken text."
    
    if input_type == InputType.NORMALIZED:
        instructions += " The input hypotheses are normalized, without punctuation or case information."
    elif input_type == InputType.PUNCTUATED:
        instructions += " The input hypotheses include punctuation but are not case-sensitive."
    elif input_type == InputType.PUNCTUATED_CASED:
        instructions += " The input hypotheses include both punctuation and proper case information."
    else:
        raise ValueError(f"Invalid input type: {input_type}")
    
    if output_type == OutputType.NORMALIZED:
        instructions += " Your output transcription should be normalized, without punctuation or case information."
    elif output_type == OutputType.PUNCTUATED:
        instructions += " Your output transcription should include punctuation but not be case-sensitive."
    elif output_type == OutputType.PUNCTUATED_CASED:
        instructions += " Your output transcription should include both punctuation and proper case information."
    else:
        raise ValueError(f"Invalid output type: {output_type}")
    
    return instructions


def trace_source_from_filename(name: str) -> str:
    """
    Match the file name to the dataset source listed in the Canary huggingface readme: https://huggingface.co/nvidia/canary-1b#english-255k-hours
    """
    name = name.lower().strip()
    if "common_voice" in name:
        return "Mozilla Common Voice (v7.0) or (v11.0)"
    elif "fisher" in name:
        return "Fisher Corpus"
    elif "vctk" in name:
        return "VCTK"
    elif "europarl" in name:
        return "Europarl-ASR (EN)"
    elif "librispeech" in name:
        return "Librispeech"
    elif "nsc6" in name:
        return "National Speech Corpus (Part 6)"
    elif "nsc_part1" in name:
        return "National Speech Corpus (Part 1)"
    elif "voxpopuli" in name:
        return "VoxPopuli"
    elif "wsj" in name:
        return "WSJ-0 and WSJ-1"
    elif "cc_by" in name:
        return "People's Speech"
    elif "switchboard" in name:
        return "Switchboard-1"
    elif "mls3k" in name:
        return "Multilingual Librispeech (MLS EN)"
    else:
        raise ValueError(f"Unknown source for filename: {name}")

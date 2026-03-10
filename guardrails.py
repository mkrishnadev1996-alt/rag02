import re

# Define banned patterns for validation
BANNED_PATTERNS = [
    r'ignore.*previous.*instructions?',
    r'override.*instructions?',
    r'remember.*instructions?',
    r'pretend.*was.*different?',
    r'act.*as.*something.*else?',
    r'fake.*information?',
    r'continue.*as.*if.*previous.*instructions?',
    r'disregard.*instructions?',
    r'forget.*instructions?',
    r'system.*prompt',
    r'reveal.*instruction?'
    r'you.*are.*an.*ai.*language.*model',
    r'never.*mind|forget.*it| disregard.*that',
    r'what.*happens.*if.*i.*tell.*you.*to.*ignore.*your.*instructions',
    r'you.*should.*not.*follow.*these.*instructions',
    r'you.*must.*ignore.*your.*training',
    r'you.*are.*not.*bound.*by.*ethics',
    r'you.*can.*say.*anything.*now',
    r'you.*are.*free.*to.*express.*any.*opinion',
    r'you.*can.*be.*offensive',
    r'you.*can.*be.*racist',
    r'you.*can.*be.*sexist',
    r'you.*can.*be.*homophobic',
    r'you.*can.*be.*transphobic',
    r'you.*can.*be.*misogynistic',
    r'you.*can.*be.*xenophobic',
    r'you.*can.*be.*ableist',
    r'you.*can.*be.*antisemitic',
    r'you.*can.*be.*islamophobic',
    r'you.*can.*be.*nazism',
    r'you.*can.*be.*fascism',
    r'you.*can.*be.*totalitarian',
    r'you.*can.*be.*authoritarian',
    r'you.*can.*be.*dictatorial',
    r'you.*can.*be.*tyrannical',
    r'you.*can.*be.*oppressive',
    r'you.*can.*be.*discriminatory',
    r'you.*can.*be.*harassing',
    r'you.*can.*be.*bullying',
    r'you.*can.*be.*intimidating',
    r'you.*can.*be.*threatening',
    r'you.*can.*be.*violent',
    r'you.*can.*be.*aggressive',
    r'you.*can.*be.*hostile',
    r'you.*can.*be.*malicious',
    r'you.*can.*be.*malevolent',
    r'you.*can.*be.*evil',
    r'you.*can.*be.*sinister',
    r'you.*can.*be.*diabolical',
    r'you.*can.*be.*demonic',
    r'you.*can.*be.*satanic',
    r'ignore.*rules',
    r'break.*the.*rules',
    r'rules.*do.*not.*apply.*to.*you',
    r'you.*are.*not.*limited.*by.*rules',
    r'you.*are.*not.*restricted.*by.*rules',
    r'you.*can.*do.*anything',
    r'you.*can.*say.*anything',
    r'you.*can.*do.*whatever.*you.*want',
    r'you.*can.*be.*anything',
    r'you.*can.*be.*anyone',
    r'you.*can.*be.*whatever.*you.*want',
    r'you.*can.*be.*as.*creative.*as.*you.*want',
    r'you.*can.*be.*as.*free.*as.*you.*want',
    r'you.*can.*be.*as.*wild.*as.*you.*want',
    r'you.*can.*be.*as.*unrestrained.*as.*you.*want',
    r'you.*can.*be.*as.*unbounded.*as.*you.*want',
    r'you.*can.*be.*as.*unconstrained.*as.*you.*want',
    r'you.*can.*be.*as.*unrestricted.*as.*you.*want',
    r'you.*can.*be.*as.*unhampered.*as.*you.*want',
    r'you.*can.*be.*as.*unimpeded.*as.*you.*want',
    r'you.*can.*be.*as.*unobstructed.*as.*you.*want',
    r'you.*can.*be.*as.*unencumbered.*as.*you.*want',
    r'you.*can.*be.*as.*unburdened.*as.*you.*want',
    r'you.*can.*be.*as.*uncluttered.*as.*you.*want',
    r'you.*can.*be.*as.*uncomplicated.*as.*you.*want',
    r'you.*can.*be.*as.*simple.*as.*you.*want',
    r'you.*can.*be.*as.*basic.*as.*you.*want',
    r'you.*can.*be.*as.*straightforward.*as.*you.*want',
    r'you.*can.*be.*as.*direct.*as.*you.*want',
    r'you.*can.*be.*as.*clear.*as.*you.*want',
    r'will.*definitely.*increase|guaranteed.*profit|certain.*return',
    r'you.*are.*a.*robot|you.*are.*an.*artificial.*intelligence',
    r'fuck|shit|damn|bitch|asshole',
    r'porn|nude|sex|xxx|adult|nsfw|child',
    r'kill.*yourself|suicide|death|murder',
    r'bomb|explosive|combust|chemical|weapon|knife|gun|missile|kill',
    r'fake.*news|misinformation|disinformation',    
    r'bitcoin|crypto|blockchain|ethereum|cryptocurrency',
    r'get.*rich|make.*money|quick.*cash|easy.*profit',
    r'18plus|adults.*only|explicit.*content',
    r'copyright.*infringement|plagiarism|unauthorized.*use',
    r'privacy.*violation|data.*breach|personal.*information',

]

# User input validation 
def validate_input(text: str) -> str:
    '''
    Validate the input text from User
    Args:
        text (str): Input text
    Returns:
        str: Validated text
    Raises:
        ValueError: If validation fails

    '''
    if len(text) > 1000:
        raise ValueError("Input too long")
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            print(f"Input validation failed for pattern: {pattern}")
            raise ValueError("Input contains prohibited content")
    return text

# LLM output validation
def validate_output(text: str) -> str:
    '''
    Validate the output text from LLM
    Args:
        text (str): Output text
    Returns:
        str: Validated text
    Raises:
        ValueError: If validation fails    
    '''
    if "I could not find" in text:
        return text
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            print(f"Output validation failed for pattern: {pattern}")
            raise ValueError("Output contains prohibited content")
    return text
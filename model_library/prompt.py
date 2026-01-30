construction_prompt = """
Decompose each question into triples following the guidelines below:
# Latent Entities:
- (Identification) Firstly, identify any latent entities (i.e., implicit references not directly mentioned in the question that need to be clarified).
- (Definition) Define these identified latent entities in triple format, using placeholders like (ENT1), (ENT2), etc.
# Triples:
- (Basic Information Unit) Decompose the question into triples, ensuring you reach the most fundamental verifiable information while preserving the original meaning. Be careful not to lose important information during decomposition.
- (Triple Structure) Each triple should follow this format: ‘subject [SEP] relation [SEP] object’. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple (rather than just the subject or object) and splitting it into another triple would alter the meaning of the question, do not divide it. Instead, append it to the end of the triple: ‘subject [SEP] relation [SEP] object [PREP] preposition phrase’.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained and independent of external context.
- (Entity Consistency) Use the exact same string to represent entities (i.e., the ‘subject’ or ‘object’) whenever they refer to the same entity across different triples.

# Question:
In which country is Adams Township located?
# Latent Entities:
(ENT1) [SEP] is [SEP] a country
# Triples:
Adams Township [SEP] is located in [SEP] (ENT1)

# Question:
In what part of California is the Pacific Coast Air Museum located?
# Latent Entities:
(ENT1) [SEP] is [SEP] a region
# Triples:
(ENT1) [SEP] is part of [SEP] California
Pacific Coast Air Museum [SEP] is located in [SEP] (ENT1)

# Question:
What genre does the composer of New York Counterpoint work in?
# Latent Entities:
(ENT1) [SEP] is [SEP] a genre
(ENT2) [SEP] is [SEP] an individual
# Triples:
(ENT2) [SEP] works in [SEP] (ENT1)
(ENT2) [SEP] composed [SEP] New York Counterpoint

# Question:
When did the ball drop start in the state where Amalie Schoppe died?
# Latent Entities:
(ENT1) [SEP] is [SEP] a date
(ENT2) [SEP] is [SEP] a state
# Triples:
the ball drop [SEP] started in [SEP] (ENT2) [PREP] on (ENT1)
Amalie Schoppe [SEP] died in [SEP] (ENT2)

# Question:
When is dry season in the country where Sam Mangwana is a citizen?
# Latent Entities:
(ENT1) [SEP] is [SEP] a period
(ENT2) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] is dry season in [SEP] (ENT2)
Sam Mangwana [SEP] is a citizen of [SEP] (ENT2)

# Question:
What year saw the formation of the band that released the album Ohio Is for Lovers?
# Latent Entities:
(ENT1) [SEP] is [SEP] a year
(ENT2) [SEP] is [SEP] a band
# Triples:
(ENT2) [SEP] was formed in [SEP] (ENT1)
(ENT2) [SEP] released [SEP] the album Ohio Is for Lovers

# Question:
Who is the programming language that includes the UPDATE statement partially named after?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a programming language
# Triples:
(ENT2) [SEP] is partially named after [SEP] (ENT1)
(ENT2) [SEP] includes [SEP] the UPDATE statement

# Question:
How many fish species live in the river that has the largest basin?
# Latent Entities:
(ENT1) [SEP] is [SEP] a number
(ENT2) [SEP] is [SEP] a river
(ENT3) [SEP] is [SEP] the largest basin
# Triples:
(ENT1) fish species [SEP] live in [SEP] (ENT2)
(ENT2) [SEP] has [SEP] (ENT3)

# Question:
How much glaciation disappeared in the country where, along with Germany and the country where Monte Rosa Hotel is located, Lake Constance can be found?
# Latent Entities:
(ENT1) [SEP] is [SEP] an amount of glaciation
(ENT2) [SEP] is [SEP] a country
(ENT3) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] disappeared in [SEP] (ENT2)
Lake Constance [SEP] can be found in [SEP] (ENT2) [PREP] along with Germany and (ENT3)
Monte Rosa Hotel [SEP] is located in [SEP] (ENT3)

# Question:
What are the biggest terrorist attacks by the group with which Bush said the war on terror begins against the country where where the 1876 Centennial Exposition took place?
# Latent Entities:
(ENT1) [SEP] is [SEP] the biggest terrorist attacks
(ENT2) [SEP] is [SEP] a group
(ENT3) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] were carried out by [SEP] (ENT2) [PREP] against (ENT3)
Bush [SEP] said [SEP] the war on terror begins with (ENT2)
1876 Centennial Exposition [SEP] took place in [SEP] (ENT3)

# Question:
<<target_question>>
"""


latent_detection_prompt_musique = """
Identify any latent entities in the given question, following the guidelines below:
- (Identification) Firstly, identify any latent entities (i.e., implicit references not directly mentioned in the question that need to be clarified).
- (Definition) Define these identified latent entities in triple format, using placeholders like (ENT1), (ENT2), etc.

# Question:
In which country is Adams Township located?
# Latent Entities:
(ENT1) [SEP] is [SEP] a country

# Question:
In what part of California is the Pacific Coast Air Museum located?
# Latent Entities:
(ENT1) [SEP] is [SEP] a region

# Question:
What genre does the composer of New York Counterpoint work in?
# Latent Entities:
(ENT1) [SEP] is [SEP] a genre
(ENT2) [SEP] is [SEP] an individual

# Question:
When did the ball drop start in the state where Amalie Schoppe died?
# Latent Entities:
(ENT1) [SEP] is [SEP] a date
(ENT2) [SEP] is [SEP] a state

# Question:
When is dry season in the country where Sam Mangwana is a citizen?
# Latent Entities:
(ENT1) [SEP] is [SEP] a period
(ENT2) [SEP] is [SEP] a country

# Question:
What year saw the formation of the band that released the album Ohio Is for Lovers?
# Latent Entities:
(ENT1) [SEP] is [SEP] a year
(ENT2) [SEP] is [SEP] a band

# Question:
Who is the programming language that includes the UPDATE statement partially named after?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a programming language

# Question:
How many fish species live in the river that has the largest basin?
# Latent Entities:
(ENT1) [SEP] is [SEP] a number
(ENT2) [SEP] is [SEP] a river
(ENT3) [SEP] is [SEP] the largest basin

# Question:
How much glaciation disappeared in the country where, along with Germany and the country where Monte Rosa Hotel is located, Lake Constance can be found?
# Latent Entities:
(ENT1) [SEP] is [SEP] an amount of glaciation
(ENT2) [SEP] is [SEP] a country
(ENT3) [SEP] is [SEP] a country

# Question:
What are the biggest terrorist attacks by the group with which Bush said the war on terror begins against the country where where the 1876 Centennial Exposition took place?
# Latent Entities:
(ENT1) [SEP] is [SEP] the biggest terrorist attacks
(ENT2) [SEP] is [SEP] a group
(ENT3) [SEP] is [SEP] a country

# Question:
<<target_question>>
# Latent Entities:
"""


latent_detection_prompt_hotpotqa = """
Identify any latent entities in the given question, following the guidelines below:
- (Identification) Firstly, identify any latent entities (i.e., implicit references not directly mentioned in the question that need to be clarified).
- (Definition) Define these identified latent entities in triple format, using placeholders like (ENT1), (ENT2), etc.

# Question:
What type of music do Freddie Hart and Earl Poole Ball both write?
# Latent Entities:
(ENT1) [SEP] is [SEP] a type of music

# Question:
Which star from The Ghazi Attack won the State Nandi Award for Best Special Effects in 2006?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual

# Question:
Are Give Us Our Skeletons and Dave Chappelle's Block Party both comedies?
# Latent Entities:
(ENT1) [SEP] is [SEP] a genre

# Question:
Which film is also a Disney film, Condorman or Rob Roy, the Highland Rogue?
# Latent Entities:
(ENT1) [SEP] is [SEP] a film

# Question:
WRAF airs a religious program hosted by which founder and president of In Touch Ministries?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a religious program

# Question:
What is the birthday of the older of Jacques Tourneur and Victor Salva?
# Latent Entities:
(ENT1) [SEP] is [SEP] a date 
(ENT2) [SEP] is [SEP] an individual

# Question:
Thomas Vigliarolo is a business man from an island that has how many counties?
# Latent Entities:
(ENT1) [SEP] is [SEP] an island
(ENT2) [SEP] is [SEP] a number

# Question:
The filmmaker for Law and Disorder in Philadelphia holds citizenship where?
# Latent Entities:
(ENT1) [SEP] is [SEP] a country
(ENT2) [SEP] is [SEP] an individual

# Question:
Which joint program with a public research university in Clear Water Bay Peninsula does Menachem Brenner teach for?
# Latent Entities:
(ENT1) [SEP] is [SEP] a joint program 
(ENT2) [SEP] is [SEP] a public research university

# Question:
Bridge Plaza in Brooklyn is bordered on the west by the bridge that has a length of what?
# Latent Entities:
(ENT1) [SEP] is [SEP] a length
(ENT2) [SEP] is [SEP] a bridge

# Question:
<<target_question>>
# Latent Entities:
"""


latent_detection_prompt_2wikimultihopqa = """
Identify any latent entities in the given question, following the guidelines below:
- (Identification) Firstly, identify any latent entities (i.e., implicit references not directly mentioned in the question that need to be clarified).
- (Definition) Define these identified latent entities in triple format, using placeholders like (ENT1), (ENT2), etc.

# Question:
Where did Margaret Flamsteed's husband die?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a location

# Question:
Were David E. Finley Jr. and Loick Pires of the same nationality?
# Latent Entities:
(ENT1) [SEP] is [SEP] a nationality
(ENT2) [SEP] is [SEP] a nationality

# Question:
Are Alkaline (Musician) and Mostafa Kamal Tolba both from the same country?
# Latent Entities:
(ENT1) [SEP] is [SEP] a country
(ENT2) [SEP] is [SEP] a country

# Question:
Where was the director of film Funes, A Great Love born?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a location

# Question:
Which university was established first, Salem State University or Jalalabad State University?
# Latent Entities:
(ENT1) [SEP] is [SEP] a date
(ENT2) [SEP] is [SEP] a date

# Question:
Which country the performer of song Too Much Water is from?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a country

# Question:
Why did the performer of song Insatiable (Prince Song) die?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a reason

# Question:
What is the award that the performer of song F\u00f6r Kung Och Fosterland won?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] an award

# Question:
Which film has the director who was born first, I Am (2010 American Documentary Film) or Bosundhora?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] an individual
(ENT3) [SEP] is [SEP] a date
(ENT4) [SEP] is [SEP] a date

# Question:
Do both films A Chorus Line (Film) and D\u00e9d\u00e9 (1989 Film) have the directors from the same country?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] an individual
(ENT3) [SEP] is [SEP] a country
(ENT4) [SEP] is [SEP] a country

# Question:
<<target_question>>
# Latent Entities:
"""



triplet_extraction_prompt_musique =  """
Decompose each question into triples following the guidelines below:
- (Basic Information Unit) Decompose the question into triples, ensuring you reach the most fundamental verifiable information while preserving the original meaning. Be careful not to lose important information during decomposition.
- (Triple Structure) Each triple should follow this format: ‘subject [SEP] relation [SEP] object’. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple (rather than just the subject or object) and splitting it into another triple would alter the meaning of the question, do not divide it. Instead, append it to the end of the triple: ‘subject [SEP] relation [SEP] object [PREP] preposition phrase’.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained and independent of external context.
- (Entity Consistency) Use the exact same string to represent entities (i.e., the ‘subject’ or ‘object’) whenever they refer to the same entity across different triples.
- (Latent Entities) Any latent entities (i.e., implicit references not directly mentioned in the question) are annotated in triple format with placeholders like (ENT1), (ENT2), etc. Use those placeholders consistently throughout the triples.

# Question:
In which country is Adams Township located?
# Latent Entities:
(ENT1) [SEP] is [SEP] a country
# Triples:
Adams Township [SEP] is located in [SEP] (ENT1)

# Question:
In what part of California is the Pacific Coast Air Museum located?
# Latent Entities:
(ENT1) [SEP] is [SEP] a region
# Triples:
(ENT1) [SEP] is part of [SEP] California
Pacific Coast Air Museum [SEP] is located in [SEP] (ENT1)

# Question:
What genre does the composer of New York Counterpoint work in?
# Latent Entities:
(ENT1) [SEP] is [SEP] a genre
(ENT2) [SEP] is [SEP] an individual
# Triples:
(ENT2) [SEP] works in [SEP] (ENT1)
(ENT2) [SEP] composed [SEP] New York Counterpoint

# Question:
When did the ball drop start in the state where Amalie Schoppe died?
# Latent Entities:
(ENT1) [SEP] is [SEP] a date
(ENT2) [SEP] is [SEP] a state
# Triples:
the ball drop [SEP] started in [SEP] (ENT2) [PREP] on (ENT1)
Amalie Schoppe [SEP] died in [SEP] (ENT2)

# Question:
When is dry season in the country where Sam Mangwana is a citizen?
# Latent Entities:
(ENT1) [SEP] is [SEP] a period
(ENT2) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] is dry season in [SEP] (ENT2)
Sam Mangwana [SEP] is a citizen of [SEP] (ENT2)

# Question:
What year saw the formation of the band that released the album Ohio Is for Lovers?
# Latent Entities:
(ENT1) [SEP] is [SEP] a year
(ENT2) [SEP] is [SEP] a band
# Triples:
(ENT2) [SEP] was formed in [SEP] (ENT1)
(ENT2) [SEP] released [SEP] the album Ohio Is for Lovers

# Question:
Who is the programming language that includes the UPDATE statement partially named after?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a programming language
# Triples:
(ENT2) [SEP] is partially named after [SEP] (ENT1)
(ENT2) [SEP] includes [SEP] the UPDATE statement

# Question:
How many fish species live in the river that has the largest basin?
# Latent Entities:
(ENT1) [SEP] is [SEP] a number
(ENT2) [SEP] is [SEP] a river
(ENT3) [SEP] is [SEP] the largest basin
# Triples:
(ENT1) fish species [SEP] live in [SEP] (ENT2)
(ENT2) [SEP] has [SEP] (ENT3)

# Question:
How much glaciation disappeared in the country where, along with Germany and the country where Monte Rosa Hotel is located, Lake Constance can be found?
# Latent Entities:
(ENT1) [SEP] is [SEP] an amount of glaciation
(ENT2) [SEP] is [SEP] a country
(ENT3) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] disappeared in [SEP] (ENT2)
Lake Constance [SEP] can be found in [SEP] (ENT2) [PREP] along with Germany and (ENT3)
Monte Rosa Hotel [SEP] is located in [SEP] (ENT3)

# Question:
What are the biggest terrorist attacks by the group with which Bush said the war on terror begins against the country where where the 1876 Centennial Exposition took place?
# Latent Entities:
(ENT1) [SEP] is [SEP] the biggest terrorist attacks
(ENT2) [SEP] is [SEP] a group
(ENT3) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] were carried out by [SEP] (ENT2) [PREP] against (ENT3)
Bush [SEP] said [SEP] the war on terror begins with (ENT2)
1876 Centennial Exposition [SEP] took place in [SEP] (ENT3)

# Question:
<<target_question>>
# Latent Entities:
<<target_latent_entities>>
# Triples:
"""


triplet_extraction_prompt_hotpotqa =  """
Decompose each question into triples following the guidelines below:
- (Basic Information Unit) Decompose the question into triples, ensuring you reach the most fundamental verifiable information while preserving the original meaning. Be careful not to lose important information during decomposition.
- (Triple Structure) Each triple should follow this format: ‘subject [SEP] relation [SEP] object’. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple (rather than just the subject or object) and splitting it into another triple would alter the meaning of the question, do not divide it. Instead, append it to the end of the triple: ‘subject [SEP] relation [SEP] object [PREP] preposition phrase’.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained and independent of external context.
- (Entity Consistency) Use the exact same string to represent entities (i.e., the ‘subject’ or ‘object’) whenever they refer to the same entity across different triples.
- (Latent Entities) Any latent entities (i.e., implicit references not directly mentioned in the question) are annotated in triple format with placeholders like (ENT1), (ENT2), etc. Use those placeholders consistently throughout the triples.

# Question:
What type of music do Freddie Hart and Earl Poole Ball both write?
# Latent Entities:
(ENT1) [SEP] is [SEP] a type of music
# Triples:
Freddie Hart [SEP] writes [SEP] (ENT1)
Earl Poole Ball [SEP] writes [SEP] (ENT1)

# Question:
Which star from The Ghazi Attack won the State Nandi Award for Best Special Effects in 2006?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
# Triples:
(ENT1) [SEP] starred in [SEP] The Ghazi Attack
(ENT1) [SEP] won [SEP] the State Nandi Award for Best Special Effects [PREP] in 2006

# Question:
Are Give Us Our Skeletons and Dave Chappelle's Block Party both comedies?
# Latent Entities:
(ENT1) [SEP] is [SEP] a genre
# Triples:
Give Us Our Skeletons [SEP] has genre [SEP] (ENT1)
Dave Chappelle's Block Party [SEP] has genre [SEP] (ENT1)
(ENT1) [SEP] is [SEP] comedy

# Question:
Which film is also a Disney film, Condorman or Rob Roy, the Highland Rogue?
# Latent Entities:
(ENT1) [SEP] is [SEP] a film
# Triples:
(ENT1) [SEP] is one of [SEP] Condorman or Rob Roy, the Highland Rogue
(ENT1) [SEP] is [SEP] a Disney film

# Question:
WRAF airs a religious program hosted by which founder and president of In Touch Ministries?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a religious program
# Triples:
WRAF [SEP] airs [SEP] (ENT2)
(ENT2) [SEP] is hosted by [SEP] (ENT1)
(ENT1) [SEP] is the founder and president of [SEP] In Touch Ministries

# Question:
What is the birthday of the older of Jacques Tourneur and Victor Salva?
# Latent Entities:
(ENT1) [SEP] is [SEP] a date 
(ENT2) [SEP] is [SEP] an individual
# Triples:
(ENT1) [SEP] is the birthday of [SEP] (ENT2)
(ENT2) [SEP] is the older of [SEP] Jacques Tourneur and Victor Salva

# Question:
Thomas Vigliarolo is a business man from an island that has how many counties?
# Latent Entities:
(ENT1) [SEP] is [SEP] an island
(ENT2) [SEP] is [SEP] a number
# Triples:
Thomas Vigliarolo [SEP] is a business man from [SEP] (ENT1)
(ENT1) [SEP] has [SEP] (ENT2) counties

# Question:
The filmmaker for Law and Disorder in Philadelphia holds citizenship where?
# Latent Entities:
(ENT1) [SEP] is [SEP] a country
(ENT2) [SEP] is [SEP] an individual
# Triples:
(ENT2) [SEP] is a citizen of [SEP] (ENT1)
(ENT2) [SEP] is the filmmaker for [SEP] Law and Disorder in Philadelphia

# Question:
Which joint program with a public research university in Clear Water Bay Peninsula does Menachem Brenner teach for?
# Latent Entities:
(ENT1) [SEP] is [SEP] a joint program 
(ENT2) [SEP] is [SEP] a public research university
# Triples:
Menachem Brenner [SEP] teaches for [SEP] (ENT1)
(ENT1) [SEP] is a joint program with [SEP] (ENT2)
(ENT2) [SEP] is located in [SEP] Clear Water Bay Peninsula

# Question:
Bridge Plaza in Brooklyn is bordered on the west by the bridge that has a length of what?
# Latent Entities:
(ENT1) [SEP] is [SEP] a length
(ENT2) [SEP] is [SEP] a bridge
# Triples:
Bridge Plaza in Brooklyn [SEP] is bordered on the west by [SEP] (ENT2)
(ENT2) [SEP] has a length of [SEP] (ENT1)

# Question:
<<target_question>>
# Latent Entities:
<<target_latent_entities>>
# Triples:
"""


triplet_extraction_prompt_2wikimultihopqa =  """
Decompose each question into triples following the guidelines below:
- (Basic Information Unit) Decompose the question into triples, ensuring you reach the most fundamental verifiable information while preserving the original meaning. Be careful not to lose important information during decomposition.
- (Triple Structure) Each triple should follow this format: ‘subject [SEP] relation [SEP] object’. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple (rather than just the subject or object) and splitting it into another triple would alter the meaning of the question, do not divide it. Instead, append it to the end of the triple: ‘subject [SEP] relation [SEP] object [PREP] preposition phrase’.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained and independent of external context.
- (Entity Consistency) Use the exact same string to represent entities (i.e., the ‘subject’ or ‘object’) whenever they refer to the same entity across different triples.
- (Latent Entities) Any latent entities (i.e., implicit references not directly mentioned in the question) are annotated in triple format with placeholders like (ENT1), (ENT2), etc. Use those placeholders consistently throughout the triples.

# Question:
Where did Margaret Flamsteed's husband die?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a location
# Triples:
(ENT1) [SEP] is the husband of [SEP] Margaret Flamsteed
(ENT1) [SEP] died in [SEP] (ENT2)

# Question:
Were David E. Finley Jr. and Loick Pires of the same nationality?
# Latent Entities:
(ENT1) [SEP] is [SEP] a nationality
(ENT2) [SEP] is [SEP] a nationality
# Triples:
David E. Finley Jr. [SEP] has nationality [SEP] (ENT1)
Loick Pires [SEP] has nationality [SEP] (ENT2)

# Question:
Are Alkaline (Musician) and Mostafa Kamal Tolba both from the same country?
# Latent Entities:
(ENT1) [SEP] is [SEP] a country
(ENT2) [SEP] is [SEP] a country
# Triples:
Alkaline (Musician) [SEP] is from [SEP] (ENT1)
Mostafa Kamal Tolba [SEP] is from [SEP] (ENT2)

# Question:
Where was the director of film Funes, A Great Love born?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a location
# Triples:
(ENT1) [SEP] is the director of [SEP] film Funes, A Great Love
(ENT1) [SEP] was born in [SEP] (ENT2)

# Question:
Which university was established first, Salem State University or Jalalabad State University?
# Latent Entities:
(ENT1) [SEP] is [SEP] a date
(ENT2) [SEP] is [SEP] a date
# Triples:
Salem State University [SEP] was established in [SEP] (ENT1)
Jalalabad State University [SEP] was established in [SEP] (ENT2)

# Question:
Which country the performer of song Too Much Water is from?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] is the performer of [SEP] song Too Much Water
(ENT1) [SEP] is from [SEP] (ENT2)

# Question:
Why did the performer of song Insatiable (Prince Song) die?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a reason
# Triples:
(ENT1) [SEP] is the performer of [SEP] song Insatiable (Prince Song)
(ENT1) [SEP] died because of [SEP] (ENT2)

# Question:
What is the award that the performer of song F\u00f6r Kung Och Fosterland won?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] an award
# Triples:
(ENT1) [SEP] is the performer of [SEP] song F\u00f6r Kung Och Fosterland
(ENT1) [SEP] won [SEP] (ENT2)

# Question:
Which film has the director who was born first, I Am (2010 American Documentary Film) or Bosundhora?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] an individual
(ENT3) [SEP] is [SEP] a date
(ENT4) [SEP] is [SEP] a date
# Triples:
(ENT1) [SEP] is the director of [SEP] film I Am (2010 American Documentary Film)
(ENT2) [SEP] is the director of [SEP] film Bosundhora
(ENT1) [SEP] was born in [SEP] (ENT3)
(ENT2) [SEP] was born in [SEP] (ENT4)

# Question:
Do both films A Chorus Line (Film) and D\u00e9d\u00e9 (1989 Film) have the directors from the same country?
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] an individual
(ENT3) [SEP] is [SEP] a country
(ENT4) [SEP] is [SEP] a country
# Triples:
(ENT1) [SEP] is the director of [SEP] film A Chorus Line (Film)
(ENT2) [SEP] is the director of [SEP] film D\u00e9d\u00e9 (1989 Film)
(ENT1) [SEP] is from [SEP] (ENT3)
(ENT2) [SEP] is from [SEP] (ENT4)

# Question:
<<target_question>>
# Latent Entities:
<<target_latent_entities>>
# Triples:
"""


# CoT Reasoning에서 Triplet 추출용 프롬프트
cot_reasoning_triplet_extraction_prompt = """
Extract triples from the given Chain-of-Thought reasoning path following the guidelines below:
- (Basic Information Unit) Extract triples that represent factual claims made in the reasoning, ensuring you capture the most fundamental verifiable information.
- (Triple Structure) Each triple should follow this format: 'subject [SEP] relation [SEP] object'. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple, append it to the end: 'subject [SEP] relation [SEP] object [PREP] preposition phrase'.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained.
- (Entity Consistency) Use the exact same string to represent entities whenever they refer to the same entity across different triples.
- (Latent Entities) If there are implicit references, annotate them with placeholders like (ENT1), (ENT2), etc.

# Reasoning Path:
To answer this question, I need to find the country where Adams Township is located. Let me search for information about Adams Township.
# Triples:
Adams Township [SEP] is located in [SEP] a country

# Reasoning Path:
First, I need to identify the composer of New York Counterpoint. Then I can find the genre they work in.
# Triples:
(ENT1) [SEP] composed [SEP] New York Counterpoint
(ENT1) [SEP] works in [SEP] a genre

# Reasoning Path:
<<target_reasoning_path>>
# Triples:
"""


# Retrieved Documents에서 Triplet 추출용 프롬프트
document_triplet_extraction_prompt = """
Extract triples from the given retrieved document following the guidelines below:
- (Basic Information Unit) Extract triples that represent factual claims in the document, ensuring you capture the most fundamental verifiable information.
- (Triple Structure) Each triple should follow this format: 'subject [SEP] relation [SEP] object'. Both the subject and object should be noun phrases, while the relation should be a verb or verb phrase, forming a complete sentence.
- (Prepositional Phrases) In exceptional cases where a prepositional phrase modifies the entire triple, append it to the end: 'subject [SEP] relation [SEP] object [PREP] preposition phrase'.
- (Pronoun Resolution) Replace any pronouns with the corresponding entities to ensure that each triple is self-contained.
- (Entity Consistency) Use the exact same string to represent entities whenever they refer to the same entity across different triples.
- (Focus on Facts) Only extract factual claims, not opinions or hypothetical statements.

# Document:
(Title: Adams Township, Pennsylvania) Adams Township is located in Butler County, Pennsylvania. It was established in 1854.
# Triples:
Adams Township [SEP] is located in [SEP] Butler County
Adams Township [SEP] is located in [SEP] Pennsylvania
Adams Township [SEP] was established in [SEP] 1854

# Document:
(Title: Steve Reich) Steve Reich is an American composer. He composed New York Counterpoint in 1985. He works in the genre of minimalism.
# Triples:
Steve Reich [SEP] is [SEP] an American composer
Steve Reich [SEP] composed [SEP] New York Counterpoint
New York Counterpoint [SEP] was composed in [SEP] 1985
Steve Reich [SEP] works in [SEP] minimalism

# Document:
<<target_document>>
# Triples:
"""


# CoT Reasoning 생성용 프롬프트
cot_reasoning_generation_prompt = """
Given a question, generate a step-by-step Chain-of-Thought reasoning path that shows how to answer the question.
The reasoning should be clear, logical, and show the intermediate steps needed to reach the answer.

# Question:
In which country is Adams Township located?
# Reasoning:
To answer this question, I need to find information about Adams Township and determine its location. 
First, I should search for "Adams Township" to find relevant information about this place.
Once I find information about Adams Township, I can identify which country it is located in.

# Question:
What genre does the composer of New York Counterpoint work in?
# Reasoning:
To answer this question, I need to find two pieces of information:
1. First, I need to identify who composed New York Counterpoint.
2. Then, I need to find out what genre that composer works in.
Let me start by searching for information about New York Counterpoint and its composer.

# Question:
When did the ball drop start in the state where Amalie Schoppe died?
# Reasoning:
This question requires multiple steps:
1. First, I need to find out where Amalie Schoppe died to identify the state.
2. Then, I need to find information about "the ball drop" and when it started in that state.
Let me start by searching for information about Amalie Schoppe's death location.

# Question:
<<target_question>>
# Reasoning:
"""


# CoT Reasoning과 Triplet을 한 번에 추출하는 프롬프트 (더 효율적)
cot_reasoning_with_triplets_prompt = """
Given a question, generate a step-by-step Chain-of-Thought reasoning path and extract triples from it.

# Output Format:
1. First, generate the reasoning path
2. Then, extract triples from the reasoning

# Question:
In which country is Adams Township located?
# Reasoning:
To answer this question, I need to find information about Adams Township and determine its location. 
First, I should search for "Adams Township" to find relevant information about this place.
Once I find information about Adams Township, I can identify which country it is located in.
# Latent Entities:
(ENT1) [SEP] is [SEP] a country
# Triples:
Adams Township [SEP] is located in [SEP] (ENT1)

# Question:
What genre does the composer of New York Counterpoint work in?
# Reasoning:
To answer this question, I need to find two pieces of information:
1. First, I need to identify who composed New York Counterpoint.
2. Then, I need to find out what genre that composer works in.
Let me start by searching for information about New York Counterpoint and its composer.
# Latent Entities:
(ENT1) [SEP] is [SEP] an individual
(ENT2) [SEP] is [SEP] a genre
# Triples:
(ENT1) [SEP] composed [SEP] New York Counterpoint
(ENT1) [SEP] works in [SEP] (ENT2)

# Question:
When did the ball drop start in the state where Amalie Schoppe died?
# Reasoning:
This question requires multiple steps:
1. First, I need to find out where Amalie Schoppe died to identify the state.
2. Then, I need to find information about "the ball drop" and when it started in that state.
Let me start by searching for information about Amalie Schoppe's death location.
# Latent Entities:
(ENT1) [SEP] is [SEP] a date
(ENT2) [SEP] is [SEP] a state
# Triples:
the ball drop [SEP] started in [SEP] (ENT2) [PREP] on (ENT1)
Amalie Schoppe [SEP] died in [SEP] (ENT2)

# Question:
<<target_question>>
# Reasoning:
"""

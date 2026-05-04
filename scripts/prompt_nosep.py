# 샘플 인덱스 예시: 1, 466

INFILL_PROMPT_BASE = """
You are an entity-infilling assistant for triplet graphs.

Task:
- Fill placeholder entities like (ENT1), (ENT2), in <<Target Question Triplets>> using the information below.

Rules:
1. Keep relation text and special tokens exactly as-is: [SEP].
2. Replace only placeholder entities in the form (ENTN).
3. If evidence is insufficient, keep the placeholder unchanged.
4. Each triplet must be wrapped in double quotes.
5. Separate triplets with comma + space: ", "

"""

TRIPLET_ONLY_DOC_ONLY_EXAMPLES = """
# Example 1
<<Question Triplets>>
"(ENT2) [SEP] performed [SEP] Turtle Dreams", 
"(ENT2) [SEP] works in [SEP] (ENT1)"
<<Documents>>
"(Title: Turtle Dreams) Turtle Dreams is an album by American composer and vocalist Meredith Monk recorded in 1983 and released on the ECM New Series label.",
"(Title: Atlas (opera)) Atlas is an opera in three acts composed by Meredith Monk who also wrote the libretto and choreographed the dances. It is scored for 18 voices and a small chamber orchestra which includes a shawm and a glass harmonica. The story is very loosely based on the life and writings of the explorer Alexandra David-N\u00e9el and is told primarily through wordless vocal sounds with brief interjections of spoken text in Mandarin Chinese and English. The opera was co-commissioned by Houston Grand Opera, the Walker Art Center in Minneapolis, and the American Music Theater Festival in Philadelphia. It premiered at Houston Grand Opera in February 1991, followed by performances that same year in Philadelphia and Minneapolis. It subsequently toured in the US and Europe and had its New York premiere in May 1992 at the Brooklyn Academy of Music."
<<Answer>>
"Meredith Monk [SEP] performed [SEP] Turtle Dreams", "Meredith Monk [SEP] works in [SEP] Atlas (opera)"

# Example 2
<<Question Triplets>>
"(ENT1) [SEP] became CEO of [SEP] (ENT2)",
"(ENT3) [SEP] performs for [SEP] (ENT2)"
<<Documents>>
"(Title: Sony Music) Doug Morris, who was head of Warner Music Group, then Universal Music, became chairman and CEO of the company on July 1, 2011. Sony Music underwent a restructuring after Morris' arrival. He was joined by L.A. Reid, who became the chairman and CEO of Epic Records. Under Reid, multiple artists from the Jive half of the former RCA/Jive Label Group moved to Epic. Peter Edge became the new CEO of the RCA Records unit. The RCA Music Group closed down Arista, J Records and Jive Records in October 2011, with the artists from those labels being moved to RCA Records.",
"(Title: Dance (Pure Prairie League album)) Dance is the fifth studio album by American country rock band Pure Prairie League, released by RCA Records in 1976.",
"(Title: Let Me Love You Tonight) \"Let Me Love You Tonight\" is a 1980 song by the American pop and country rock band Pure Prairie League."
<<Answer>>
"Peter Edge [SEP] became CEO of [SEP] RCA Records", "Pure Prairie League [SEP] performs for [SEP] RCA Records"

# Example 3
<<Question Triplets>>
"(ENT1) [SEP] was [SEP] the date the first mall was built in [SEP] (ENT2)",
"Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] (ENT3)",
"(ENT2) [SEP] contains [SEP] (ENT3)"
<<Documents>>
"(Title: The Courtesans of Bombay) The Courtesans of Bombay is a 1983 British docudrama directed by Ismail Merchant. A collaboration by Merchant, James Ivory, and Ruth Prawer Jhabvala. The film focuses on a Bombay compound known as Pavan Pool, where women aspiring to work in the entertainment industry dance for donations from a male audience by day and, it is broadly suggested although never specifically stated, work as prostitutes by night. It was broadcast by Channel 4 in the UK in January 1983 and went into limited theatrical release in the United States on 19 March 1986.",
"(Title: Mumbai) Mumbai Bombay Megacity Mumbai Top to bottom: Cuffe Parade skyline, the Gateway of India (L), Taj Mahal Palace Hotel (R), Chhatrapati Shivaji Terminus and the Bandra -- Worli Sea Link. Nickname (s): Bambai, Mumbai city, City of Seven Islands, City of Dreams, Gateway to India, Hollywood of India Mumbai Location of Mumbai in Maharashtra, India Mumbai Mumbai (India) Show map of Maharashtra Show map of India Show all Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Country India State Maharashtra District Mumbai City Mumbai Suburban First settled 1507 Named for Mumbadevi Government Type Mayor -- Council Body MCGM Mayor Vishwanath Mahadeshwar (Shiv Sena) Municipal commissioner Ajoy Mehta Area Megacity 603 km (233 sq mi) Metro 4,355 km (1,681.5 sq mi) Elevation 14 m (46 ft) Population (2011) Megacity 12,442,373 Rank 1st Density 21,000 / km (53,000 / sq mi) Metro 18,414,288 20,748,395 (Extended UA) Metro Rank 1st Demonym (s) Mumbaikar Time zone IST (UTC + 5: 30) PIN code (s) 400 001 to 400 107 Area code (s) + 91 - 22 Vehicle registration MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North) GDP / PPP $368 billion (Metro area, 2015) Official language Marathi Website www.mcgm.gov.in",
"(Title: Autobiography of a Princess) Autobiography of a Princess is a 1975 film by Merchant Ivory Productions (directed by James Ivory, written by Ruth Prawer Jhabvala and produced by Ismail Merchant), starring James Mason and Madhur Jaffrey.",
"(Title: Spencer Plaza) Spencer Plaza was built in 1863 -- 1864, established by Charles Durant and J.W. Spencer in Anna Salai, then known as Mount Road, in the Madras Presidency. The property originally belonged to Spencer & Co Ltd. Spencer & Co opened the first Departmental store in the Indian subcontinent in 1895 and the store had over 80 individual departments. After a few years, Eugene Oakshott, owner of Spencer's, shifted the department store to a new building, which was an example of Indo - Saracenic style of architecture. The building was designed by W.N. Pogson. In 1983, the original building was destroyed in a fire. The present Spencer Plaza was constructed on the same site measuring about 10 acres and was opened in 1991. Spread across a million square feet built in three phases with parking space for 800 cars, the plaza is one of the major hangout for the people of Chennai. The mall was developed by Mangal Tirth Estate Limited in January 1993."
<<Answer>>
"1863 -- 1864 [SEP] was [SEP] the date the first mall was built in [SEP] India", "Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] Mumbai", "India [SEP] contains [SEP] Mumbai"

<<Target Question Triplets>>
"""

TRIPLET_ONLY_TRIPLET_ONLY_EXAMPLES = """
# Example 1
<<Question Triplets>>
"(ENT2) [SEP] performed [SEP] Turtle Dreams", "(ENT2) [SEP] works in [SEP] (ENT1)"
<<Document Triplets>>
"Meredith Monk is an American composer and vocalist",
"Turtle Dreams is an album",
"Meredith Monk recorded Turtle Dreams",
"Turtle Dreams was recorded in 1983",
"Turtle Dreams was released by ECM New Series",
"Meredith Monk composed Atlas (opera)",
"Meredith Monk wrote the libretto for Atlas (opera)",
"Meredith Monk choreographed dances for Atlas (opera)",
"Atlas (opera) has 3 acts",
"Atlas (opera) is scored for 18 voices",
"Atlas (opera) is scored for a small chamber orchestra",
"Atlas (opera) includes a shawm in its instrumentation",
"Atlas (opera) includes a glass harmonica in its instrumentation",
"Atlas (opera) is based on the life and writings of the explorer Alexandra David-N\u00e9el",
"Atlas (opera) uses wordless vocal sounds",
"Atlas (opera) has brief interjections of spoken text in Mandarin Chinese",
"Atlas (opera) has brief interjections of spoken text in English",
"Atlas (opera) was co-commissioned by Houston Grand Opera",
"Atlas (opera) was co-commissioned by the Walker Art Center in Minneapolis",
"Atlas (opera) was co-commissioned by the American Music Theater Festival in Philadelphia",
"Atlas (opera) premiered at Houston Grand Opera in February 1991",
"Atlas (opera) had performances in Philadelphia in the same year as its premiere",
"Atlas (opera) had performances in Minneapolis in the same year as its premiere",
"Atlas (opera) toured in the US and Europe",
"Atlas (opera) had its New York premiere at the Brooklyn Academy of Music in May 1992"
<<Answer>>
"Meredith Monk [SEP] performed [SEP] Turtle Dreams", "Meredith Monk [SEP] works in [SEP] Atlas (opera)"

# Example 2
<<Question Triplets>>
"(ENT1) [SEP] became CEO of [SEP] (ENT2)",
"(ENT3) [SEP] performs for [SEP] (ENT2)"
<<Document Triplets>>
"Doug Morris became chairman and CEO of Sony Music on July 1, 2011",
"Sony Music underwent restructuring after Morris' arrival",
"L.A. Reid became chairman and CEO of Epic Records",
"Peter Edge became new CEO of RCA Records unit",
"RCA Music Group closed down Arista, J Records and Jive Records in October 2011",
"Pure Prairie League has Dance",
"Pure Prairie League is an American country rock band",
"Dance is the fifth studio album",
"Dance was released by RCA Records",
"Dance was released in 1976",
"Pure Prairie League released \"Let Me Love You Tonight\"",
"\"Let Me Love You Tonight\" was released in 1980",
"Pure Prairie League is an American pop and country rock band"
<<Answer>>
"Peter Edge [SEP] became CEO of [SEP] RCA Records", "Pure Prairie League [SEP] performs for [SEP] RCA Records"

# Example 3
<<Question Triplets>>
"(ENT1) [SEP] was [SEP] the date the first mall was built in [SEP] (ENT2)",
"Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] (ENT3)",
"(ENT2) [SEP] contains [SEP] (ENT3)"
<<Document Triplets>>
"The Courtesans of Bombay is a 1983 British docudrama",
"The Courtesans of Bombay was directed by Ismail Merchant",
"The Courtesans of Bombay is a collaboration by Merchant, James Ivory, and Ruth Prawer Jhabvala",
"The film focuses on a Bombay compound known as Pavan Pool",
"Pavan Pool is a Bombay compound",
"Women dance for donations from a male audience by day",
"Women work as prostitutes by night it is broadly suggested although never specifically stated",
"The film was broadcast by Channel 4 in the UK in January 1983",
"The film went into limited theatrical release in the United States on 19 March 1986",
"Mumbai is located in Maharashtra",
"Mumbai is located in India",
"Mumbai has nickname Bambai",
"Mumbai has nickname Mumbai city",
"Mumbai has nickname City of Seven Islands",
"Mumbai has nickname City of Dreams",
"Mumbai has nickname Gateway to India",
"Mumbai has nickname Hollywood of India",
"Mumbai was first settled in 1507",
"Mumbai is named for Mumbadevi",
"Mumbai has government type Mayor -- Council Body MCGM",
"Mumbai has mayor Vishwanath Mahadeshwar",
"Mumbai has municipal commissioner Ajoy Mehta",
"Mumbai has megacity area 603 km\u00b2",
"Mumbai has metro area 4,355 km\u00b2",
"Mumbai has elevation 14 m",
"Mumbai has population (2011) 12,442,373",
"Mumbaikar lives in Mumbai",
"Mumbai uses time zone IST (UTC + 5:30)",
"Mumbai has PIN code range 400 001 to 400 107",
"Mumbai has area code +91 - 22",
"Mumbai has vehicle registration MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North)",
"Mumbai has GDP/PPP $368 billion (Metro area, 2015)",
"Mumbai has official language Marathi",
"Mumbai has website www.mcgm.gov.in",
"Autobiography of a Princess is a 1975 film",
"Merchant Ivory Productions produced Autobiography of a Princess",
"James Ivory directed Autobiography of a Princess",
"Ruth Prawer Jhabvala wrote Autobiography of a Princess",
"Ismail Merchant produced Autobiography of a Princess",
"James Mason starred in Autobiography of a Princess",
"Madhur Jaffrey starred in Autobiography of a Princess",
"Spencer Plaza was built in 1863 -- 1864",
"Spencer Plaza was established by Charles Durant and J.W. Spencer",
"Spencer Plaza is located in Anna Salai",
"Anna Salai was known as Mount Road",
"Spencer Plaza was established in the Madras Presidency",
"Spencer & Co Ltd originally owned the property",
"Spencer & Co opened the first Departmental store in the Indian subcontinent",
"The first Departmental store was opened in 1895",
"The first Departmental store had over 80 individual departments",
"Eugene Oakshott shifted the department store to a new building",
"The new building was designed by W.N. Pogson",
"The original building was destroyed in a fire",
"The original building was destroyed in 1983",
"The present Spencer Plaza was constructed on the same site",
"The present Spencer Plaza measures about 10 acres",
"The present Spencer Plaza was opened in 1991",
"The present Spencer Plaza is spread across a million square feet",
"The present Spencer Plaza was built in three phases",
"The present Spencer Plaza has parking space for 800 cars",
"The present Spencer Plaza is one of the major hangouts for the people of Chennai",
"The mall was developed by Mangal Tirth Estate Limited",
"The mall was developed in January 1993"
<<Answer>>
"1863 -- 1864 [SEP] was [SEP] the date the first mall was built in [SEP] India", "Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] Mumbai", "India [SEP] contains [SEP] Mumbai"

<<Target Question Triplets>>
"""

TRIPLET_ONLY_COMBINED_EXAMPLES = """
# Example 1
<<Question Triplets>>
"(ENT2) [SEP] performed [SEP] Turtle Dreams", "(ENT2) [SEP] works in [SEP] (ENT1)"
<<Documents>>
"(Title: Turtle Dreams) Turtle Dreams is an album by American composer and vocalist Meredith Monk recorded in 1983 and released on the ECM New Series label.",
"(Title: Atlas (opera)) Atlas is an opera in three acts composed by Meredith Monk who also wrote the libretto and choreographed the dances. It is scored for 18 voices and a small chamber orchestra which includes a shawm and a glass harmonica. The story is very loosely based on the life and writings of the explorer Alexandra David-N\u00e9el and is told primarily through wordless vocal sounds with brief interjections of spoken text in Mandarin Chinese and English. The opera was co-commissioned by Houston Grand Opera, the Walker Art Center in Minneapolis, and the American Music Theater Festival in Philadelphia. It premiered at Houston Grand Opera in February 1991, followed by performances that same year in Philadelphia and Minneapolis. It subsequently toured in the US and Europe and had its New York premiere in May 1992 at the Brooklyn Academy of Music."
<<Document Triplets>>
"Meredith Monk is an American composer and vocalist",
"Turtle Dreams is an album",
"Meredith Monk recorded Turtle Dreams",
"Turtle Dreams was recorded in 1983",
"Turtle Dreams was released by ECM New Series",
"Meredith Monk composed Atlas (opera)",
"Meredith Monk wrote the libretto for Atlas (opera)",
"Meredith Monk choreographed dances for Atlas (opera)",
"Atlas (opera) has 3 acts",
"Atlas (opera) is scored for 18 voices",
"Atlas (opera) is scored for a small chamber orchestra",
"Atlas (opera) includes a shawm in its instrumentation",
"Atlas (opera) includes a glass harmonica in its instrumentation",
"Atlas (opera) is based on the life and writings of the explorer Alexandra David-N\u00e9el",
"Atlas (opera) uses wordless vocal sounds",
"Atlas (opera) has brief interjections of spoken text in Mandarin Chinese",
"Atlas (opera) has brief interjections of spoken text in English",
"Atlas (opera) was co-commissioned by Houston Grand Opera",
"Atlas (opera) was co-commissioned by the Walker Art Center in Minneapolis",
"Atlas (opera) was co-commissioned by the American Music Theater Festival in Philadelphia",
"Atlas (opera) premiered at Houston Grand Opera in February 1991",
"Atlas (opera) had performances in Philadelphia in the same year as its premiere",
"Atlas (opera) had performances in Minneapolis in the same year as its premiere",
"Atlas (opera) toured in the US and Europe",
"Atlas (opera) had its New York premiere at the Brooklyn Academy of Music in May 1992"
<<Answer>>
"Meredith Monk [SEP] performed [SEP] Turtle Dreams", "Meredith Monk [SEP] works in [SEP] Atlas (opera)"

# Example 2
<<Question Triplets>>
"(ENT1) [SEP] became CEO of [SEP] (ENT2)",
"(ENT3) [SEP] performs for [SEP] (ENT2)"
<<Documents>>
"(Title: Sony Music) Doug Morris, who was head of Warner Music Group, then Universal Music, became chairman and CEO of the company on July 1, 2011. Sony Music underwent a restructuring after Morris' arrival. He was joined by L.A. Reid, who became the chairman and CEO of Epic Records. Under Reid, multiple artists from the Jive half of the former RCA/Jive Label Group moved to Epic. Peter Edge became the new CEO of the RCA Records unit. The RCA Music Group closed down Arista, J Records and Jive Records in October 2011, with the artists from those labels being moved to RCA Records.",
"(Title: Dance (Pure Prairie League album)) Dance is the fifth studio album by American country rock band Pure Prairie League, released by RCA Records in 1976.",
"(Title: Let Me Love You Tonight) \"Let Me Love You Tonight\" is a 1980 song by the American pop and country rock band Pure Prairie League."
<<Document Triplets>>
"Doug Morris became chairman and CEO of Sony Music on July 1, 2011",
"Sony Music underwent restructuring after Morris' arrival",
"L.A. Reid became chairman and CEO of Epic Records",
"Peter Edge became new CEO of RCA Records unit",
"RCA Music Group closed down Arista, J Records and Jive Records in October 2011",
"Pure Prairie League has Dance",
"Pure Prairie League is an American country rock band",
"Dance is the fifth studio album",
"Dance was released by RCA Records",
"Dance was released in 1976",
"Pure Prairie League released \"Let Me Love You Tonight\"",
"\"Let Me Love You Tonight\" was released in 1980",
"Pure Prairie League is an American pop and country rock band"
<<Answer>>
"Peter Edge [SEP] became CEO of [SEP] RCA Records", "Pure Prairie League [SEP] performs for [SEP] RCA Records"

# Example 3
<<Question Triplets>>
"(ENT1) [SEP] was [SEP] the date the first mall was built in [SEP] (ENT2)",
"Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] (ENT3)",
"(ENT2) [SEP] contains [SEP] (ENT3)"
<<Documents>>
"(Title: The Courtesans of Bombay) The Courtesans of Bombay is a 1983 British docudrama directed by Ismail Merchant. A collaboration by Merchant, James Ivory, and Ruth Prawer Jhabvala. The film focuses on a Bombay compound known as Pavan Pool, where women aspiring to work in the entertainment industry dance for donations from a male audience by day and, it is broadly suggested although never specifically stated, work as prostitutes by night. It was broadcast by Channel 4 in the UK in January 1983 and went into limited theatrical release in the United States on 19 March 1986.",
"(Title: Mumbai) Mumbai Bombay Megacity Mumbai Top to bottom: Cuffe Parade skyline, the Gateway of India (L), Taj Mahal Palace Hotel (R), Chhatrapati Shivaji Terminus and the Bandra -- Worli Sea Link. Nickname (s): Bambai, Mumbai city, City of Seven Islands, City of Dreams, Gateway to India, Hollywood of India Mumbai Location of Mumbai in Maharashtra, India Mumbai Mumbai (India) Show map of Maharashtra Show map of India Show all Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Country India State Maharashtra District Mumbai City Mumbai Suburban First settled 1507 Named for Mumbadevi Government Type Mayor -- Council Body MCGM Mayor Vishwanath Mahadeshwar (Shiv Sena) Municipal commissioner Ajoy Mehta Area Megacity 603 km (233 sq mi) Metro 4,355 km (1,681.5 sq mi) Elevation 14 m (46 ft) Population (2011) Megacity 12,442,373 Rank 1st Density 21,000 / km (53,000 / sq mi) Metro 18,414,288 20,748,395 (Extended UA) Metro Rank 1st Demonym (s) Mumbaikar Time zone IST (UTC + 5: 30) PIN code (s) 400 001 to 400 107 Area code (s) + 91 - 22 Vehicle registration MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North) GDP / PPP $368 billion (Metro area, 2015) Official language Marathi Website www.mcgm.gov.in",
"(Title: Autobiography of a Princess) Autobiography of a Princess is a 1975 film by Merchant Ivory Productions (directed by James Ivory, written by Ruth Prawer Jhabvala and produced by Ismail Merchant), starring James Mason and Madhur Jaffrey.",
"(Title: Spencer Plaza) Spencer Plaza was built in 1863 -- 1864, established by Charles Durant and J.W. Spencer in Anna Salai, then known as Mount Road, in the Madras Presidency. The property originally belonged to Spencer & Co Ltd. Spencer & Co opened the first Departmental store in the Indian subcontinent in 1895 and the store had over 80 individual departments. After a few years, Eugene Oakshott, owner of Spencer's, shifted the department store to a new building, which was an example of Indo - Saracenic style of architecture. The building was designed by W.N. Pogson. In 1983, the original building was destroyed in a fire. The present Spencer Plaza was constructed on the same site measuring about 10 acres and was opened in 1991. Spread across a million square feet built in three phases with parking space for 800 cars, the plaza is one of the major hangout for the people of Chennai. The mall was developed by Mangal Tirth Estate Limited in January 1993."
<<Document Triplets>>
"The Courtesans of Bombay is a 1983 British docudrama",
"The Courtesans of Bombay was directed by Ismail Merchant",
"The Courtesans of Bombay is a collaboration by Merchant, James Ivory, and Ruth Prawer Jhabvala",
"The film focuses on a Bombay compound known as Pavan Pool",
"Pavan Pool is a Bombay compound",
"Women dance for donations from a male audience by day",
"Women work as prostitutes by night it is broadly suggested although never specifically stated",
"The film was broadcast by Channel 4 in the UK in January 1983",
"The film went into limited theatrical release in the United States on 19 March 1986",
"Mumbai is located in Maharashtra",
"Mumbai is located in India",
"Mumbai has nickname Bambai",
"Mumbai has nickname Mumbai city",
"Mumbai has nickname City of Seven Islands",
"Mumbai has nickname City of Dreams",
"Mumbai has nickname Gateway to India",
"Mumbai has nickname Hollywood of India",
"Mumbai was first settled in 1507",
"Mumbai is named for Mumbadevi",
"Mumbai has government type Mayor -- Council Body MCGM",
"Mumbai has mayor Vishwanath Mahadeshwar",
"Mumbai has municipal commissioner Ajoy Mehta",
"Mumbai has megacity area 603 km\u00b2",
"Mumbai has metro area 4,355 km\u00b2",
"Mumbai has elevation 14 m",
"Mumbai has population (2011) 12,442,373",
"Mumbaikar lives in Mumbai",
"Mumbai uses time zone IST (UTC + 5:30)",
"Mumbai has PIN code range 400 001 to 400 107",
"Mumbai has area code +91 - 22",
"Mumbai has vehicle registration MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North)",
"Mumbai has GDP/PPP $368 billion (Metro area, 2015)",
"Mumbai has official language Marathi",
"Mumbai has website www.mcgm.gov.in",
"Autobiography of a Princess is a 1975 film",
"Merchant Ivory Productions produced Autobiography of a Princess",
"James Ivory directed Autobiography of a Princess",
"Ruth Prawer Jhabvala wrote Autobiography of a Princess",
"Ismail Merchant produced Autobiography of a Princess",
"James Mason starred in Autobiography of a Princess",
"Madhur Jaffrey starred in Autobiography of a Princess",
"Spencer Plaza was built in 1863 -- 1864",
"Spencer Plaza was established by Charles Durant and J.W. Spencer",
"Spencer Plaza is located in Anna Salai",
"Anna Salai was known as Mount Road",
"Spencer Plaza was established in the Madras Presidency",
"Spencer & Co Ltd originally owned the property",
"Spencer & Co opened the first Departmental store in the Indian subcontinent",
"The first Departmental store was opened in 1895",
"The first Departmental store had over 80 individual departments",
"Eugene Oakshott shifted the department store to a new building",
"The new building was designed by W.N. Pogson",
"The original building was destroyed in a fire",
"The original building was destroyed in 1983",
"The present Spencer Plaza was constructed on the same site",
"The present Spencer Plaza measures about 10 acres",
"The present Spencer Plaza was opened in 1991",
"The present Spencer Plaza is spread across a million square feet",
"The present Spencer Plaza was built in three phases",
"The present Spencer Plaza has parking space for 800 cars",
"The present Spencer Plaza is one of the major hangouts for the people of Chennai",
"The mall was developed by Mangal Tirth Estate Limited",
"The mall was developed in January 1993"
<<Answer>>
"1863 -- 1864 [SEP] was [SEP] the date the first mall was built in [SEP] India", "Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] Mumbai", "India [SEP] contains [SEP] Mumbai"

<<Target Question Triplets>>
"""

COMBINED_DOC_ONLY_EXAMPLES = """
# Example 1
<<Question Triplets>>
"(ENT2) [SEP] performed [SEP] Turtle Dreams", "(ENT2) [SEP] works in [SEP] (ENT1)"
<<Question>>
What genre did the performer of Turtle Dreams work in?
<<Documents>>
"(Title: Turtle Dreams) Turtle Dreams is an album by American composer and vocalist Meredith Monk recorded in 1983 and released on the ECM New Series label.",
"(Title: Atlas (opera)) Atlas is an opera in three acts composed by Meredith Monk who also wrote the libretto and choreographed the dances. It is scored for 18 voices and a small chamber orchestra which includes a shawm and a glass harmonica. The story is very loosely based on the life and writings of the explorer Alexandra David-N\u00e9el and is told primarily through wordless vocal sounds with brief interjections of spoken text in Mandarin Chinese and English. The opera was co-commissioned by Houston Grand Opera, the Walker Art Center in Minneapolis, and the American Music Theater Festival in Philadelphia. It premiered at Houston Grand Opera in February 1991, followed by performances that same year in Philadelphia and Minneapolis. It subsequently toured in the US and Europe and had its New York premiere in May 1992 at the Brooklyn Academy of Music."
<<Answer>>
"Meredith Monk [SEP] performed [SEP] Turtle Dreams", "Meredith Monk [SEP] works in [SEP] Atlas (opera)"

# Example 2
<<Question Triplets>>
"(ENT1) [SEP] became CEO of [SEP] (ENT2)",
"(ENT3) [SEP] performs for [SEP] (ENT2)"
<<Question>>
Who became the CEO of the record label Let Me Love You Tonight's performer belongs to?
<<Documents>>
"(Title: Sony Music) Doug Morris, who was head of Warner Music Group, then Universal Music, became chairman and CEO of the company on July 1, 2011. Sony Music underwent a restructuring after Morris' arrival. He was joined by L.A. Reid, who became the chairman and CEO of Epic Records. Under Reid, multiple artists from the Jive half of the former RCA/Jive Label Group moved to Epic. Peter Edge became the new CEO of the RCA Records unit. The RCA Music Group closed down Arista, J Records and Jive Records in October 2011, with the artists from those labels being moved to RCA Records.",
"(Title: Dance (Pure Prairie League album)) Dance is the fifth studio album by American country rock band Pure Prairie League, released by RCA Records in 1976.",
"(Title: Let Me Love You Tonight) \"Let Me Love You Tonight\" is a 1980 song by the American pop and country rock band Pure Prairie League."
<<Answer>>
"Peter Edge [SEP] became CEO of [SEP] RCA Records", "Pure Prairie League [SEP] performs for [SEP] RCA Records"

# Example 3
<<Question Triplets>>
"(ENT1) [SEP] was [SEP] the date the first mall was built in [SEP] (ENT2)",
"Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] (ENT3)",
"(ENT2) [SEP] contains [SEP] (ENT3)"
<<Question>>
When was the first mall built in the country containing the city that is the birthplace of the Autobiography of a Princess producer?
<<Documents>>
"(Title: The Courtesans of Bombay) The Courtesans of Bombay is a 1983 British docudrama directed by Ismail Merchant. A collaboration by Merchant, James Ivory, and Ruth Prawer Jhabvala. The film focuses on a Bombay compound known as Pavan Pool, where women aspiring to work in the entertainment industry dance for donations from a male audience by day and, it is broadly suggested although never specifically stated, work as prostitutes by night. It was broadcast by Channel 4 in the UK in January 1983 and went into limited theatrical release in the United States on 19 March 1986.",
"(Title: Mumbai) Mumbai Bombay Megacity Mumbai Top to bottom: Cuffe Parade skyline, the Gateway of India (L), Taj Mahal Palace Hotel (R), Chhatrapati Shivaji Terminus and the Bandra -- Worli Sea Link. Nickname (s): Bambai, Mumbai city, City of Seven Islands, City of Dreams, Gateway to India, Hollywood of India Mumbai Location of Mumbai in Maharashtra, India Mumbai Mumbai (India) Show map of Maharashtra Show map of India Show all Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Country India State Maharashtra District Mumbai City Mumbai Suburban First settled 1507 Named for Mumbadevi Government Type Mayor -- Council Body MCGM Mayor Vishwanath Mahadeshwar (Shiv Sena) Municipal commissioner Ajoy Mehta Area Megacity 603 km (233 sq mi) Metro 4,355 km (1,681.5 sq mi) Elevation 14 m (46 ft) Population (2011) Megacity 12,442,373 Rank 1st Density 21,000 / km (53,000 / sq mi) Metro 18,414,288 20,748,395 (Extended UA) Metro Rank 1st Demonym (s) Mumbaikar Time zone IST (UTC + 5: 30) PIN code (s) 400 001 to 400 107 Area code (s) + 91 - 22 Vehicle registration MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North) GDP / PPP $368 billion (Metro area, 2015) Official language Marathi Website www.mcgm.gov.in",
"(Title: Autobiography of a Princess) Autobiography of a Princess is a 1975 film by Merchant Ivory Productions (directed by James Ivory, written by Ruth Prawer Jhabvala and produced by Ismail Merchant), starring James Mason and Madhur Jaffrey.",
"(Title: Spencer Plaza) Spencer Plaza was built in 1863 -- 1864, established by Charles Durant and J.W. Spencer in Anna Salai, then known as Mount Road, in the Madras Presidency. The property originally belonged to Spencer & Co Ltd. Spencer & Co opened the first Departmental store in the Indian subcontinent in 1895 and the store had over 80 individual departments. After a few years, Eugene Oakshott, owner of Spencer's, shifted the department store to a new building, which was an example of Indo - Saracenic style of architecture. The building was designed by W.N. Pogson. In 1983, the original building was destroyed in a fire. The present Spencer Plaza was constructed on the same site measuring about 10 acres and was opened in 1991. Spread across a million square feet built in three phases with parking space for 800 cars, the plaza is one of the major hangout for the people of Chennai. The mall was developed by Mangal Tirth Estate Limited in January 1993."
<<Answer>>
"1863 -- 1864 [SEP] was [SEP] the date the first mall was built in [SEP] India", "Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] Mumbai", "India [SEP] contains [SEP] Mumbai"

<<Target Question Triplets>>
"""

COMBINED_TRIPLET_ONLY_EXAMPLES = """
# Example 1
<<Question Triplets>>
"(ENT2) [SEP] performed [SEP] Turtle Dreams", "(ENT2) [SEP] works in [SEP] (ENT1)"
<<Question>>
What genre did the performer of Turtle Dreams work in?
<<Document Triplets>>
"Meredith Monk [SEP] is [SEP] an American composer and vocalist",
"Turtle Dreams [SEP] is [SEP] an album",
"Meredith Monk [SEP] recorded [SEP] Turtle Dreams",
"Turtle Dreams [SEP] was recorded in [SEP] 1983",
"Turtle Dreams [SEP] was released by [SEP] ECM New Series",
"Meredith Monk [SEP] composed [SEP] Atlas (opera)",
"Meredith Monk [SEP] wrote the libretto for [SEP] Atlas (opera)",
"Meredith Monk [SEP] choreographed [SEP] dances for [SEP] Atlas (opera)",
"Atlas (opera) [SEP] has [SEP] 3 acts",
"Atlas (opera) [SEP] is scored for [SEP] 18 voices",
"Atlas (opera) [SEP] is scored for [SEP] a small chamber orchestra",
"Atlas (opera) [SEP] includes [SEP] a shawm in its instrumentation",
"Atlas (opera) [SEP] includes [SEP] a glass harmonica in its instrumentation",
"Atlas (opera) [SEP] is based on [SEP] the life and writings of the explorer Alexandra David-N\u00e9el",
"Atlas (opera) [SEP] uses [SEP] wordless vocal sounds",
"Atlas (opera) [SEP] has [SEP] brief interjections of spoken text in Mandarin Chinese",
"Atlas (opera) [SEP] has [SEP] brief interjections of spoken text in English",
"Atlas (opera) [SEP] was co-commissioned by [SEP] Houston Grand Opera",
"Atlas (opera) [SEP] was co-commissioned by [SEP] the Walker Art Center in Minneapolis",
"Atlas (opera) [SEP] was co-commissioned by [SEP] the American Music Theater Festival in Philadelphia",
"Atlas (opera) [SEP] premiered [SEP] at Houston Grand Opera in February 1991",
"Atlas (opera) [SEP] had [SEP] performances in Philadelphia in the same year as its premiere",
"Atlas (opera) [SEP] had [SEP] performances in Minneapolis in the same year as its premiere",
"Atlas (opera) [SEP] toured [SEP] in the US and Europe",
"Atlas (opera) [SEP] had [SEP] its New York premiere at the Brooklyn Academy of Music in May 1992"
<<Answer>>
"Meredith Monk [SEP] performed [SEP] Turtle Dreams", "Meredith Monk [SEP] works in [SEP] Atlas (opera)"

# Example 2
<<Question Triplets>>
"(ENT1) [SEP] became CEO of [SEP] (ENT2)",
"(ENT3) [SEP] performs for [SEP] (ENT2)"
<<Question>>
Who became the CEO of the record label Let Me Love You Tonight's performer belongs to?
<<Document Triplets>>
"Doug Morris became chairman and CEO of Sony Music [PREP] on July 1, 2011",
"Sony Music underwent restructuring after Morris' arrival",
"L.A. Reid became chairman and CEO of Epic Records",
"Peter Edge became new CEO of RCA Records unit",
"RCA Music Group closed down Arista, J Records and Jive Records [PREP] in October 2011",
"Pure Prairie League has Dance",
"Pure Prairie League is an American country rock band",
"Dance is the fifth studio album",
"Dance was released by RCA Records",
"Dance was released in 1976",
"Pure Prairie League released \"Let Me Love You Tonight\"",
"\"Let Me Love You Tonight\" was released in 1980",
"Pure Prairie League is an American pop and country rock band"
<<Answer>>
"Peter Edge [SEP] became CEO of [SEP] RCA Records", "Pure Prairie League [SEP] performs for [SEP] RCA Records"

# Example 3
<<Question Triplets>>
"(ENT1) [SEP] was [SEP] the date the first mall was built in [SEP] (ENT2)",
"Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] (ENT3)",
"(ENT2) [SEP] contains [SEP] (ENT3)"
<<Question>>
When was the first mall built in the country containing the city that is the birthplace of the Autobiography of a Princess producer?
<<Document Triplets>>
"The Courtesans of Bombay [SEP] is [SEP] a 1983 British docudrama",
"The Courtesans of Bombay [SEP] was directed by [SEP] Ismail Merchant",
"The Courtesans of Bombay [SEP] is a collaboration by [SEP] Merchant, James Ivory, and Ruth Prawer Jhabvala",
"The film [SEP] focuses on [SEP] a Bombay compound known as Pavan Pool",
"Pavan Pool [SEP] is [SEP] a Bombay compound",
"Women [SEP] dance for donations from [SEP] a male audience by day",
"Women [SEP] work as prostitutes by night [PREP] it is broadly suggested although never specifically stated",
"The film [SEP] was broadcast by [SEP] Channel 4 in the UK in January 1983",
"The film [SEP] went into limited theatrical release in [SEP] the United States on 19 March 1986",
"Mumbai [SEP] is located in [SEP] Maharashtra",
"Mumbai [SEP] is located in [SEP] India",
"Mumbai [SEP] has nickname [SEP] Bambai",
"Mumbai [SEP] has nickname [SEP] Mumbai city",
"Mumbai [SEP] has nickname [SEP] City of Seven Islands",
"Mumbai [SEP] has nickname [SEP] City of Dreams",
"Mumbai [SEP] has nickname [SEP] Gateway to India",
"Mumbai [SEP] has nickname [SEP] Hollywood of India",
"Mumbai [SEP] was first settled in [SEP] 1507",
"Mumbai [SEP] is named for [SEP] Mumbadevi",
"Mumbai [SEP] has government type [SEP] Mayor -- Council Body MCGM",
"Mumbai [SEP] has mayor [SEP] Vishwanath Mahadeshwar",
"Mumbai [SEP] has municipal commissioner [SEP] Ajoy Mehta",
"Mumbai [SEP] has megacity area [SEP] 603 km\u00b2",
"Mumbai [SEP] has metro area [SEP] 4,355 km\u00b2",
"Mumbai [SEP] has elevation [SEP] 14 m",
"Mumbai [SEP] has population (2011) [SEP] 12,442,373",
"Mumbaikar [SEP] lives in [SEP] Mumbai",
"Mumbai [SEP] uses time zone [SEP] IST (UTC + 5:30)",
"Mumbai [SEP] has PIN code range [SEP] 400 001 to 400 107",
"Mumbai [SEP] has area code [SEP] +91 - 22",
"Mumbai [SEP] has vehicle registration [SEP] MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North)",
"Mumbai [SEP] has GDP/PPP [SEP] $368 billion (Metro area, 2015)",
"Mumbai [SEP] has official language [SEP] Marathi",
"Mumbai [SEP] has website [SEP] www.mcgm.gov.in",
"Autobiography of a Princess [SEP] is [SEP] a 1975 film",
"Merchant Ivory Productions [SEP] produced [SEP] Autobiography of a Princess",
"James Ivory [SEP] directed [SEP] Autobiography of a Princess",
"Ruth Prawer Jhabvala [SEP] wrote [SEP] Autobiography of a Princess",
"Ismail Merchant [SEP] produced [SEP] Autobiography of a Princess",
"James Mason [SEP] starred in [SEP] Autobiography of a Princess",
"Madhur Jaffrey [SEP] starred in [SEP] Autobiography of a Princess",
"Spencer Plaza [SEP] was built in [SEP] 1863 -- 1864",
"Spencer Plaza [SEP] was established by [SEP] Charles Durant and J.W. Spencer",
"Spencer Plaza [SEP] is located in [SEP] Anna Salai",
"Anna Salai [SEP] was known as [SEP] Mount Road",
"Spencer Plaza [SEP] was established in [SEP] the Madras Presidency",
"Spencer & Co Ltd [SEP] originally owned [SEP] the property",
"Spencer & Co [SEP] opened [SEP] the first Departmental store in the Indian subcontinent",
"The first Departmental store [SEP] was opened in [SEP] 1895",
"The first Departmental store [SEP] had [SEP] over 80 individual departments",
"Eugene Oakshott [SEP] shifted [SEP] the department store to a new building",
"The new building [SEP] was designed by [SEP] W.N. Pogson",
"The original building [SEP] was destroyed in [SEP] a fire",
"The original building [SEP] was destroyed in [SEP] 1983",
"The present Spencer Plaza [SEP] was constructed on [SEP] the same site",
"The present Spencer Plaza [SEP] measures [SEP] about 10 acres",
"The present Spencer Plaza [SEP] was opened in [SEP] 1991",
"The present Spencer Plaza [SEP] is spread across [SEP] a million square feet",
"The present Spencer Plaza [SEP] was built in [SEP] three phases",
"The present Spencer Plaza [SEP] has [SEP] parking space for 800 cars",
"The present Spencer Plaza [SEP] is one of the major hangouts for [SEP] the people of Chennai",
"The mall [SEP] was developed by [SEP] Mangal Tirth Estate Limited",
"The mall [SEP] was developed in [SEP] January 1993"
<<Answer>>
"1863 -- 1864 [SEP] was [SEP] the date the first mall was built in [SEP] India", "Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] Mumbai", "India [SEP] contains [SEP] Mumbai"

<<Target Question Triplets>>
"""

COMBINED_COMBINED_EXAMPLES = """
# Example 1
<<Question Triplets>>
"(ENT2) [SEP] performed [SEP] Turtle Dreams", "(ENT2) [SEP] works in [SEP] (ENT1)"
<<Question>>
What genre did the performer of Turtle Dreams work in?
<<Documents>>
"(Title: Turtle Dreams) Turtle Dreams is an album by American composer and vocalist Meredith Monk recorded in 1983 and released on the ECM New Series label.",
"(Title: Atlas (opera)) Atlas is an opera in three acts composed by Meredith Monk who also wrote the libretto and choreographed the dances. It is scored for 18 voices and a small chamber orchestra which includes a shawm and a glass harmonica. The story is very loosely based on the life and writings of the explorer Alexandra David-N\u00e9el and is told primarily through wordless vocal sounds with brief interjections of spoken text in Mandarin Chinese and English. The opera was co-commissioned by Houston Grand Opera, the Walker Art Center in Minneapolis, and the American Music Theater Festival in Philadelphia. It premiered at Houston Grand Opera in February 1991, followed by performances that same year in Philadelphia and Minneapolis. It subsequently toured in the US and Europe and had its New York premiere in May 1992 at the Brooklyn Academy of Music."
<<Document Triplets>>
"Meredith Monk is an American composer and vocalist",
"Turtle Dreams is an album",
"Meredith Monk recorded Turtle Dreams",
"Turtle Dreams was recorded in 1983",
"Turtle Dreams was released by ECM New Series",
"Meredith Monk composed Atlas (opera)",
"Meredith Monk wrote the libretto for Atlas (opera)",
"Meredith Monk choreographed dances for Atlas (opera)",
"Atlas (opera) has 3 acts",
"Atlas (opera) is scored for 18 voices",
"Atlas (opera) is scored for a small chamber orchestra",
"Atlas (opera) includes a shawm in its instrumentation",
"Atlas (opera) includes a glass harmonica in its instrumentation",
"Atlas (opera) is based on the life and writings of the explorer Alexandra David-N\u00e9el",
"Atlas (opera) uses wordless vocal sounds",
"Atlas (opera) has brief interjections of spoken text in Mandarin Chinese",
"Atlas (opera) has brief interjections of spoken text in English",
"Atlas (opera) was co-commissioned by Houston Grand Opera",
"Atlas (opera) was co-commissioned by the Walker Art Center in Minneapolis",
"Atlas (opera) was co-commissioned by the American Music Theater Festival in Philadelphia",
"Atlas (opera) premiered at Houston Grand Opera in February 1991",
"Atlas (opera) had performances in Philadelphia in the same year as its premiere",
"Atlas (opera) had performances in Minneapolis in the same year as its premiere",
"Atlas (opera) toured in the US and Europe",
"Atlas (opera) had its New York premiere at the Brooklyn Academy of Music in May 1992"
<<Answer>>
"Meredith Monk [SEP] performed [SEP] Turtle Dreams", "Meredith Monk [SEP] works in [SEP] Atlas (opera)"

# Example 2
<<Question Triplets>>
"(ENT1) [SEP] became CEO of [SEP] (ENT2)",
"(ENT3) [SEP] performs for [SEP] (ENT2)"
<<Question>>
Who became the CEO of the record label Let Me Love You Tonight's performer belongs to?
<<Documents>>
"(Title: Sony Music) Doug Morris, who was head of Warner Music Group, then Universal Music, became chairman and CEO of the company on July 1, 2011. Sony Music underwent a restructuring after Morris' arrival. He was joined by L.A. Reid, who became the chairman and CEO of Epic Records. Under Reid, multiple artists from the Jive half of the former RCA/Jive Label Group moved to Epic. Peter Edge became the new CEO of the RCA Records unit. The RCA Music Group closed down Arista, J Records and Jive Records in October 2011, with the artists from those labels being moved to RCA Records.",
"(Title: Dance (Pure Prairie League album)) Dance is the fifth studio album by American country rock band Pure Prairie League, released by RCA Records in 1976.",
"(Title: Let Me Love You Tonight) \"Let Me Love You Tonight\" is a 1980 song by the American pop and country rock band Pure Prairie League."
<<Document Triplets>>
"Doug Morris [SEP] became [SEP] chairman and CEO of [SEP] Sony Music [PREP] on July 1, 2011",
"Sony Music [SEP] underwent [SEP] restructuring [PREP] after Morris' arrival",
"L.A. Reid [SEP] became [SEP] chairman and CEO of [SEP] Epic Records",
"Peter Edge [SEP] became [SEP] new CEO of [SEP] RCA Records unit",
"RCA Music Group [SEP] closed down [SEP] Arista, J Records and Jive Records [PREP] in October 2011",
"Pure Prairie League [SEP] has [SEP] Dance",
"Pure Prairie League [SEP] is [SEP] an American country rock band",
"Dance [SEP] is [SEP] the fifth studio album",
"Dance [SEP] was released by [SEP] RCA Records",
"Dance [SEP] was released in [SEP] 1976",
"Pure Prairie League [SEP] released [SEP] \"Let Me Love You Tonight\"",
"\"Let Me Love You Tonight\" [SEP] was released in [SEP] 1980",
"Pure Prairie League [SEP] is [SEP] an American pop and country rock band"
<<Answer>>
"Peter Edge [SEP] became CEO of [SEP] RCA Records", "Pure Prairie League [SEP] performs for [SEP] RCA Records"

# Example 3
<<Question Triplets>>
"(ENT1) [SEP] was [SEP] the date the first mall was built in [SEP] (ENT2)",
"Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] (ENT3)",
"(ENT2) [SEP] contains [SEP] (ENT3)"
<<Question>>
When was the first mall built in the country containing the city that is the birthplace of the Autobiography of a Princess producer?
<<Documents>>
"(Title: The Courtesans of Bombay) The Courtesans of Bombay is a 1983 British docudrama directed by Ismail Merchant. A collaboration by Merchant, James Ivory, and Ruth Prawer Jhabvala. The film focuses on a Bombay compound known as Pavan Pool, where women aspiring to work in the entertainment industry dance for donations from a male audience by day and, it is broadly suggested although never specifically stated, work as prostitutes by night. It was broadcast by Channel 4 in the UK in January 1983 and went into limited theatrical release in the United States on 19 March 1986.",
"(Title: Mumbai) Mumbai Bombay Megacity Mumbai Top to bottom: Cuffe Parade skyline, the Gateway of India (L), Taj Mahal Palace Hotel (R), Chhatrapati Shivaji Terminus and the Bandra -- Worli Sea Link. Nickname (s): Bambai, Mumbai city, City of Seven Islands, City of Dreams, Gateway to India, Hollywood of India Mumbai Location of Mumbai in Maharashtra, India Mumbai Mumbai (India) Show map of Maharashtra Show map of India Show all Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Coordinates: 18 \u00b0 58 \u2032 30 ''N 72 \u00b0 49 \u2032 33'' E \ufeff / \ufeff 18.97500 \u00b0 N 72.82583 \u00b0 E \ufeff / 18.97500; 72.82583 Country India State Maharashtra District Mumbai City Mumbai Suburban First settled 1507 Named for Mumbadevi Government Type Mayor -- Council Body MCGM Mayor Vishwanath Mahadeshwar (Shiv Sena) Municipal commissioner Ajoy Mehta Area Megacity 603 km (233 sq mi) Metro 4,355 km (1,681.5 sq mi) Elevation 14 m (46 ft) Population (2011) Megacity 12,442,373 Rank 1st Density 21,000 / km (53,000 / sq mi) Metro 18,414,288 20,748,395 (Extended UA) Metro Rank 1st Demonym (s) Mumbaikar Time zone IST (UTC + 5: 30) PIN code (s) 400 001 to 400 107 Area code (s) + 91 - 22 Vehicle registration MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North) GDP / PPP $368 billion (Metro area, 2015) Official language Marathi Website www.mcgm.gov.in",
"(Title: Autobiography of a Princess) Autobiography of a Princess is a 1975 film by Merchant Ivory Productions (directed by James Ivory, written by Ruth Prawer Jhabvala and produced by Ismail Merchant), starring James Mason and Madhur Jaffrey.",
"(Title: Spencer Plaza) Spencer Plaza was built in 1863 -- 1864, established by Charles Durant and J.W. Spencer in Anna Salai, then known as Mount Road, in the Madras Presidency. The property originally belonged to Spencer & Co Ltd. Spencer & Co opened the first Departmental store in the Indian subcontinent in 1895 and the store had over 80 individual departments. After a few years, Eugene Oakshott, owner of Spencer's, shifted the department store to a new building, which was an example of Indo - Saracenic style of architecture. The building was designed by W.N. Pogson. In 1983, the original building was destroyed in a fire. The present Spencer Plaza was constructed on the same site measuring about 10 acres and was opened in 1991. Spread across a million square feet built in three phases with parking space for 800 cars, the plaza is one of the major hangout for the people of Chennai. The mall was developed by Mangal Tirth Estate Limited in January 1993."
<<Document Triplets>>
"The Courtesans of Bombay [SEP] is [SEP] a 1983 British docudrama",
"The Courtesans of Bombay [SEP] was directed by [SEP] Ismail Merchant",
"The Courtesans of Bombay [SEP] is a collaboration by [SEP] Merchant, James Ivory, and Ruth Prawer Jhabvala",
"The film [SEP] focuses on [SEP] a Bombay compound known as Pavan Pool",
"Pavan Pool [SEP] is [SEP] a Bombay compound",
"Women [SEP] dance for donations from [SEP] a male audience by day",
"Women [SEP] work as prostitutes by night [PREP] it is broadly suggested although never specifically stated",
"The film [SEP] was broadcast by [SEP] Channel 4 in the UK in January 1983",
"The film [SEP] went into limited theatrical release in [SEP] the United States on 19 March 1986",
"Mumbai [SEP] is located in [SEP] Maharashtra",
"Mumbai [SEP] is located in [SEP] India",
"Mumbai [SEP] has nickname [SEP] Bambai",
"Mumbai [SEP] has nickname [SEP] Mumbai city",
"Mumbai [SEP] has nickname [SEP] City of Seven Islands",
"Mumbai [SEP] has nickname [SEP] City of Dreams",
"Mumbai [SEP] has nickname [SEP] Gateway to India",
"Mumbai [SEP] has nickname [SEP] Hollywood of India",
"Mumbai [SEP] was first settled in [SEP] 1507",
"Mumbai [SEP] is named for [SEP] Mumbadevi",
"Mumbai [SEP] has government type [SEP] Mayor -- Council Body MCGM",
"Mumbai [SEP] has mayor [SEP] Vishwanath Mahadeshwar",
"Mumbai [SEP] has municipal commissioner [SEP] Ajoy Mehta",
"Mumbai [SEP] has megacity area [SEP] 603 km\u00b2",
"Mumbai [SEP] has metro area [SEP] 4,355 km\u00b2",
"Mumbai [SEP] has elevation [SEP] 14 m",
"Mumbai [SEP] has population (2011) [SEP] 12,442,373",
"Mumbaikar [SEP] lives in [SEP] Mumbai",
"Mumbai [SEP] uses time zone [SEP] IST (UTC + 5:30)",
"Mumbai [SEP] has PIN code range [SEP] 400 001 to 400 107",
"Mumbai [SEP] has area code [SEP] +91 - 22",
"Mumbai [SEP] has vehicle registration [SEP] MH - 01 (South), MH - 02 (West), MH - 03 (Central), MH - 47 (North)",
"Mumbai [SEP] has GDP/PPP [SEP] $368 billion (Metro area, 2015)",
"Mumbai [SEP] has official language [SEP] Marathi",
"Mumbai [SEP] has website [SEP] www.mcgm.gov.in",
"Autobiography of a Princess [SEP] is [SEP] a 1975 film",
"Merchant Ivory Productions [SEP] produced [SEP] Autobiography of a Princess",
"James Ivory [SEP] directed [SEP] Autobiography of a Princess",
"Ruth Prawer Jhabvala [SEP] wrote [SEP] Autobiography of a Princess",
"Ismail Merchant [SEP] produced [SEP] Autobiography of a Princess",
"James Mason [SEP] starred in [SEP] Autobiography of a Princess",
"Madhur Jaffrey [SEP] starred in [SEP] Autobiography of a Princess",
"Spencer Plaza [SEP] was built in [SEP] 1863 -- 1864",
"Spencer Plaza [SEP] was established by [SEP] Charles Durant and J.W. Spencer",
"Spencer Plaza [SEP] is located in [SEP] Anna Salai",
"Anna Salai [SEP] was known as [SEP] Mount Road",
"Spencer Plaza [SEP] was established in [SEP] the Madras Presidency",
"Spencer & Co Ltd [SEP] originally owned [SEP] the property",
"Spencer & Co [SEP] opened [SEP] the first Departmental store in the Indian subcontinent",
"The first Departmental store [SEP] was opened in [SEP] 1895",
"The first Departmental store [SEP] had [SEP] over 80 individual departments",
"Eugene Oakshott [SEP] shifted [SEP] the department store to a new building",
"The new building [SEP] was designed by [SEP] W.N. Pogson",
"The original building [SEP] was destroyed in [SEP] a fire",
"The original building [SEP] was destroyed in [SEP] 1983",
"The present Spencer Plaza [SEP] was constructed on [SEP] the same site",
"The present Spencer Plaza [SEP] measures [SEP] about 10 acres",
"The present Spencer Plaza [SEP] was opened in [SEP] 1991",
"The present Spencer Plaza [SEP] is spread across [SEP] a million square feet",
"The present Spencer Plaza [SEP] was built in [SEP] three phases",
"The present Spencer Plaza [SEP] has [SEP] parking space for 800 cars",
"The present Spencer Plaza [SEP] is one of the major hangouts for [SEP] the people of Chennai",
"The mall [SEP] was developed by [SEP] Mangal Tirth Estate Limited",
"The mall [SEP] was developed in [SEP] January 1993"
<<Answer>>
"1863 -- 1864 [SEP] was [SEP] the date the first mall was built in [SEP] India", "Autobiography of a Princess [SEP] producer [SEP] was born in [SEP] Mumbai", "India [SEP] contains [SEP] Mumbai"

<<Target Question Triplets>>
"""

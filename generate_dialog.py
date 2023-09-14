from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import json
import random
from tqdm import tqdm


LLAMA_CHECKPOINT_DIR = ""  # Please fill in the llama checkpoint directory
COLLECTION_JSONL = "datasets/or-quac/collection.jsonl"
N_PASSAGES = 10000

examples_first_turn = [
    """Title: Drake (musician)
Context: Drake planned to release his debut album, Thank Me Later, in late 2008, but the album's release date was postponed, first to March 2010, and then to May 25, 2010. Young Money and Universal Motown had then released a statement that the album had again been pushed back three weeks, for a June 15, 2010, release. On March 9, 2010, Drake released the debut single "Over", peaking at number fourteen on the Billboard Hot 100, as well as topping the Rap Songs chart. It also received a nomination for Best Rap Solo Performance at the 53rd Grammy Awards.
Question: What is Thank Me Later in regards to Drake?""",
    """Title: The Sweet
Context: The Sweet's first full LP album, Funny How Sweet Co-Co Can Be, was released in November 1971. A collection of the band's recent singles supplemented by some new Chinn/Chapman tunes (including "Chop Chop" and "Tom Tom Turnaround") and pop covers (such as the Lovin' Spoonful's "Daydream" and the Supremes' "Reflections"), the album, recorded at Nova Studios in London, was produced by Phil Wainman and engineered by Richard Dodd and Eric Holland. It was not a serious contender on the charts.
Question: What was the name of the band The Sweet's first album?""",
    """Title: Sue Townsend
Context: Her new partner encouraged her to join a writers' group at the Phoenix Theatre, Leicester, in 1978, when she was in her early thirties. Initially too shy to speak, she did not write anything for six weeks, but was then given a fortnight to write a play. This became the thirty-minute drama Womberang (1979), set in the waiting room of a gynaecology department. At the Phoenix, she became the writer-in-residence.
Question: When did Sue Townsend transition to a writing career?""",
    """Title: Etruscan civilization
Context: Meanwhile, Rome had started annexing Etruscan cities. This led to the loss of the northern Etruscan provinces. Etruria was conquered by Rome in the 3rd century BC. Etruscan art was produced by the Etruscan civilization between the 9th and 2nd centuries BC. Particularly strong in this tradition were figurative sculpture in terracotta (particularly lifesize on sarcophagi or temples), wall-painting and metalworking (especially engraved bronze mirrors).
Question: What kind of art was the Etruscan known for?""",
    """Title: Doug Flutie
Context: Flutie played football for Boston College, the only Division I-A school to recruit him, from 1981 to 1984, and won the Heisman Trophy, Maxwell Award, and the Davey O'Brien National Quarterback Award in his senior year (1984). Flutie became the first quarterback to win the Heisman since Pat Sullivan in 1971. Flutie left school as the NCAA's all-time passing yardage leader with 10,579 yards and was a consensus All-American as a senior. He earned Player of the Year awards from UPI, Kodak, The Sporting News, and the Maxwell Football Club.
Question: What position did Doug Flutie play in college?""",
    """Title: Art Donovan
Context: He published an autobiography, Fatso, in 1987. He was noted as a jovial and humorous person during his playing career and capitalized on that with television and speaking appearances after retiring as a player. He owned and managed a country club near Baltimore. Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other "old school" footballers he played with and against.
Question: What did Art Donovan do when Donovan left the NFL?"""
]

template_first_turn = """Write a question that has an answer in the context.

{examples}

Title: {title}
Context: {context}
Question:"""

examples_follow_up = [
    {
        "title": "Drake (musician)",
        "contexts": [
            "Drake planned to release his debut album, Thank Me Later, in late 2008, but the album's release date was postponed, first to March 2010, and then to May 25, 2010. Young Money and Universal Motown had then released a statement that the album had again been pushed back three weeks, for a June 15, 2010, release. On March 9, 2010, Drake released the debut single \"Over\", peaking at number fourteen on the Billboard Hot 100, as well as topping the Rap Songs chart. It also received a nomination for Best Rap Solo Performance at the 53rd Grammy Awards.",
            "The third single and fourth singles, \"Miss Me\" and \"Fancy\" respectively, attained moderate commercial success, however, the latter garnered Drake his second nomination at the 53rd Grammy Awards, for Best Rap Performance by a Duo or Group. On April 29, it was reportedly announced that Drake had finished Thank Me Later during a show in Kansas City, Missouri. Thank Me Later was released on June 15, 2010, debuting at number one on the Billboard 200 with sales of over 447,000 copies in its first week.",
            "The third single and fourth singles, \"Miss Me\" and \"Fancy\" respectively, attained moderate commercial success, however, the latter garnered Drake his second nomination at the 53rd Grammy Awards, for Best Rap Performance by a Duo or Group. On April 29, it was reportedly announced that Drake had finished Thank Me Later during a show in Kansas City, Missouri. Thank Me Later was released on June 15, 2010, debuting at number one on the Billboard 200 with sales of over 447,000 copies in its first week.",
            "Drake planned to release his debut album, Thank Me Later, in late 2008, but the album's release date was postponed, first to March 2010, and then to May 25, 2010. Young Money and Universal Motown had then released a statement that the album had again been pushed back three weeks, for a June 15, 2010, release. On March 9, 2010, Drake released the debut single \"Over\", peaking at number fourteen on the Billboard Hot 100, as well as topping the Rap Songs chart. It also received a nomination for Best Rap Solo Performance at the 53rd Grammy Awards.",
            "It was soon announced that Drake would have a prominent role in military science fiction video game, Gears of War 3. He was scheduled to play the part of Jace Stratton, but scheduling conflicts with his upcoming Away from Home Tour prevented Drake from accepting the role. He began the tour on September 20, 2010, in Miami, Florida, performing at 78 shows over four different legs. It concluded in Las Vegas in November 2010. Due to the success of the Away from Home Tour, Drake hosted the first OVO Festival in 2010.",
            "The album became the top selling debut album for any artist in 2010, and featured Lil Wayne, Kanye West, and Jay Z. It was soon announced that Drake would have a prominent role in military science fiction video game, Gears of War 3. He was scheduled to play the part of Jace Stratton, but scheduling conflicts with his upcoming Away from Home Tour prevented Drake from accepting the role. He began the tour on September 20, 2010, in Miami, Florida, performing at 78 shows over four different legs.",
            "Due to the success of the Away from Home Tour, Drake hosted the first OVO Festival in 2010. It would soon become a regular event during the summer, with the Molson Amphitheatre in Toronto playing host to the festival on its annual cycle. Drake also had an eco-friendly college tour to support the album, beginning with Eastern Illinois University in Charleston, Illinois. It concluded in Plymouth, New Hampshire on May 8, and he had also performed at The Bamboozle on May 1.",
            "He began the tour on September 20, 2010, in Miami, Florida, performing at 78 shows over four different legs. It concluded in Las Vegas in November 2010. Due to the success of the Away from Home Tour, Drake hosted the first OVO Festival in 2010. It would soon become a regular event during the summer, with the Molson Amphitheatre in Toronto playing host to the festival on its annual cycle.",
            "The album became the top selling debut album for any artist in 2010, and featured Lil Wayne, Kanye West, and Jay Z. It was soon announced that Drake would have a prominent role in military science fiction video game, Gears of War 3. He was scheduled to play the part of Jace Stratton, but scheduling conflicts with his upcoming Away from Home Tour prevented Drake from accepting the role.",
            "Upon the album's release, 25,000 fans gathered at New York City's South Street Seaport for a free concert, hosted by Drake and Hanson, which was later cancelled by police after a near-riot ensued due to overflowing crowds. The album became the top selling debut album for any artist in 2010, and featured Lil Wayne, Kanye West, and Jay Z. It was soon announced that Drake would have a prominent role in military science fiction video game, Gears of War 3.",
            "Drake planned to release his debut album, Thank Me Later, in late 2008, but the album's release date was postponed, first to March 2010, and then to May 25, 2010. Young Money and Universal Motown had then released a statement that the album had again been pushed back three weeks, for a June 15, 2010, release. On March 9, 2010, Drake released the debut single \"Over\", peaking at number fourteen on the Billboard Hot 100, as well as topping the Rap Songs chart. It also received a nomination for Best Rap Solo Performance at the 53rd Grammy Awards."
        ],
        "questions": [
            "What is Thank Me Later in regards to Drake?",
            "How did it do?",
            "Did he tour during this time?",
            "Did he win any awards?",
            "Did he tour in any other countries?",
            "Did he make any television appearances?",
            "Are there any other interesting aspects about this article?",
            "When did this happen?",
            "What did he have to say about it?",
            "What was most prevalent in this article?",
            "Which album is that?"
        ]
    },
    {
        "title": "The Sweet",
        "contexts": [
            "The Sweet's first full LP album, Funny How Sweet Co-Co Can Be, was released in November 1971. A collection of the band's recent singles supplemented by some new Chinn/Chapman tunes (including \"Chop Chop\" and \"Tom Tom Turnaround\") and pop covers (such as the Lovin' Spoonful's \"Daydream\" and the Supremes' \"Reflections\"), the album, recorded at Nova Studios in London, was produced by Phil Wainman and engineered by Richard Dodd and Eric Holland. It was not a serious contender on the charts.",
            "The Sweet's first full LP album, Funny How Sweet Co-Co Can Be, was released in November 1971. A collection of the band's recent singles supplemented by some new Chinn/Chapman tunes (including \"Chop Chop\" and \"Tom Tom Turnaround\") and pop covers (such as the Lovin' Spoonful's \"Daydream\" and the Supremes' \"Reflections\"), the album, recorded at Nova Studios in London, was produced by Phil Wainman and engineered by Richard Dodd and Eric Holland. It was not a serious contender on the charts.",
            "The Sweet's first full LP album, Funny How Sweet Co-Co Can Be, was released in November 1971. A collection of the band's recent singles supplemented by some new Chinn/Chapman tunes (including \"Chop Chop\" and \"Tom Tom Turnaround\") and pop covers (such as the Lovin' Spoonful's \"Daydream\" and the Supremes' \"Reflections\"), the album, recorded at Nova Studios in London, was produced by Phil Wainman and engineered by Richard Dodd and Eric Holland. It was not a serious contender on the charts.",
            "In March 1971 RCA issued \"Funny Funny\", written by Chinn and Chapman, which became the group's first international hit, climbing to the Top 20 on many of the world's charts. EMI reissued their 1970 single \"All You'll Ever Get from Me\" (May 1971) and it again failed to chart. Their next RCA release \"Co-Co\" (June 1971) went to number two in the U.K. and their follow up single, \"Alexander Graham Bell\" (October 1971), only went to #33. These tracks still featured session musicians on the instruments with the quartet providing only the vocals.",
            "The Sweet's first full LP album, Funny How Sweet Co-Co Can Be, was released in November 1971. A collection of the band's recent singles supplemented by some new Chinn/Chapman tunes (including \"Chop Chop\" and \"Tom Tom Turnaround\") and pop covers (such as the Lovin' Spoonful's \"Daydream\" and the Supremes' \"Reflections\"), the album, recorded at Nova Studios in London, was produced by Phil Wainman and engineered by Richard Dodd and Eric Holland. It was not a serious contender on the charts.",
            "The Sweet's first full LP album, Funny How Sweet Co-Co Can Be, was released in November 1971. A collection of the band's recent singles supplemented by some new Chinn/Chapman tunes (including \"Chop Chop\" and \"Tom Tom Turnaround\") and pop covers (such as the Lovin' Spoonful's \"Daydream\" and the Supremes' \"Reflections\"), the album, recorded at Nova Studios in London, was produced by Phil Wainman and engineered by Richard Dodd and Eric Holland. It was not a serious contender on the charts.",
            "The Sweet made their UK television debut in December 1970 on a pop show called Lift Off, performing the song \"Funny Funny\". A management deal was signed with the aforementioned songwriting team of Nicky Chinn and Mike Chapman. Phil Wainman resumed his collaboration with Sweet, as executive producer. This management deal also included a worldwide (the U.S. excepted) record contract with RCA Records (in the United States and Canada Bell Records issued the group's music until late 1973; followed by Capitol Records).",
            "In March 1971 RCA issued \"Funny Funny\", written by Chinn and Chapman, which became the group's first international hit, climbing to the Top 20 on many of the world's charts. EMI reissued their 1970 single \"All You'll Ever Get from Me\" (May 1971) and it again failed to chart. Their next RCA release \"Co-Co\" (June 1971) went to number two in the U.K. and their follow up single, \"Alexander Graham Bell\" (October 1971), only went to #33. These tracks still featured session musicians on the instruments with the quartet providing only the vocals.",
        ],
        "questions": [
            "What was the name of the band The Sweet's first album?",
            "Who produced their first album?",
            "What year was their first album released?",
            "Was there a popular single on the album?",
            "Where was the album recorded?",
            "Did the album reach the top of the music charts?",
            "Was their album only released in the UK?",
            "What was the most popular song?"
        ]
    },
    {
        "title": "Sue Townsend",
        "contexts": [
            "Her new partner encouraged her to join a writers' group at the Phoenix Theatre, Leicester, in 1978, when she was in her early thirties. Initially too shy to speak, she did not write anything for six weeks, but was then given a fortnight to write a play. This became the thirty-minute drama Womberang (1979), set in the waiting room of a gynaecology department. At the Phoenix, she became the writer-in-residence.",
            "Her new partner encouraged her to join a writers' group at the Phoenix Theatre, Leicester, in 1978, when she was in her early thirties. Initially too shy to speak, she did not write anything for six weeks, but was then given a fortnight to write a play. This became the thirty-minute drama Womberang (1979), set in the waiting room of a gynaecology department. At the Phoenix, she became the writer-in-residence.",
            "Her new partner encouraged her to join a writers' group at the Phoenix Theatre, Leicester, in 1978, when she was in her early thirties. Initially too shy to speak, she did not write anything for six weeks, but was then given a fortnight to write a play. This became the thirty-minute drama Womberang (1979), set in the waiting room of a gynaecology department. At the Phoenix, she became the writer-in-residence.",
            "Her new partner encouraged her to join a writers' group at the Phoenix Theatre, Leicester, in 1978, when she was in her early thirties. Initially too shy to speak, she did not write anything for six weeks, but was then given a fortnight to write a play. This became the thirty-minute drama Womberang (1979), set in the waiting room of a gynaecology department. At the Phoenix, she became the writer-in-residence.",
            "Her new partner encouraged her to join a writers' group at the Phoenix Theatre, Leicester, in 1978, when she was in her early thirties. Initially too shy to speak, she did not write anything for six weeks, but was then given a fortnight to write a play. This became the thirty-minute drama Womberang (1979), set in the waiting room of a gynaecology department. At the Phoenix, she became the writer-in-residence.",
            "Her new partner encouraged her to join a writers' group at the Phoenix Theatre, Leicester, in 1978, when she was in her early thirties. Initially too shy to speak, she did not write anything for six weeks, but was then given a fortnight to write a play. This became the thirty-minute drama Womberang (1979), set in the waiting room of a gynaecology department. At the Phoenix, she became the writer-in-residence.",
            "During this time she was mentored by several theatre directors including Ian Giles and principally Sue Pomeroy who commissioned and directed a number of her plays including Womberang, Dayroom, Groping for Words and subsequently Ear, Nose and Throat. She was also introduced to William Ash, then chairman of the Soho Poly (now Soho Theatre), who likewise played a significant part in shaping her early career.",
            "During this time she was mentored by several theatre directors including Ian Giles and principally Sue Pomeroy who commissioned and directed a number of her plays including Womberang, Dayroom, Groping for Words and subsequently Ear, Nose and Throat. She was also introduced to William Ash, then chairman of the Soho Poly (now Soho Theatre), who likewise played a significant part in shaping her early career.",
            "She met writer-director Carole Hayman on the stairs of the Soho Poly theatre and went on to develop many theatre pieces with her for the Royal Court and Joint Stock, including Bazarre and Rummage and The Great Celestial Cow. They later co-wrote two television series, The Refuge and The Spinney. At the time of writing the first Adrian Mole book, Townsend was living on the Eyres Monsell Estate, near the house in which playwright Joe Orton was brought up."
        ],
        "questions": [
            "When did Sue Townsend transition to a writing career?",
            "Was this a good experience?",
            "Did she write a play during this time?",
            "Was this one of her most popular writings?",
            "When did she start writing full-time?",
            "Did she do well financially?",
            "Are there any other interesting aspects about this article?",
            "What other plays did she write?",
            "Were the productions of her plays sold out?"
        ]
    },
    {
        "title": "Etruscan civilization",
        "contexts": [
            "Meanwhile, Rome had started annexing Etruscan cities. This led to the loss of the northern Etruscan provinces. Etruria was conquered by Rome in the 3rd century BC. Etruscan art was produced by the Etruscan civilization between the 9th and 2nd centuries BC. Particularly strong in this tradition were figurative sculpture in terracotta (particularly lifesize on sarcophagi or temples), wall-painting and metalworking (especially engraved bronze mirrors).",
            "Meanwhile, Rome had started annexing Etruscan cities. This led to the loss of the northern Etruscan provinces. Etruria was conquered by Rome in the 3rd century BC. Etruscan art was produced by the Etruscan civilization between the 9th and 2nd centuries BC. Particularly strong in this tradition were figurative sculpture in terracotta (particularly lifesize on sarcophagi or temples), wall-painting and metalworking (especially engraved bronze mirrors).",
            "Most surviving Etruscan art comes from tombs, including all the fresco wall-paintings, which show scenes of feasting and some narrative mythological subjects. Bucchero wares in black were the early and native styles of fine Etruscan pottery. There was also a tradition of elaborate Etruscan vase painting, which sprung from its Greek equivalent; the Etruscans were the main export market for Greek vases.",
            "Etruscan art was strongly connected to religion; the afterlife was of major importance in Etruscan art. The Etruscan musical instruments seen in frescoes and bas-reliefs are different types of pipes, such as the plagiaulos (the pipes of Pan or Syrinx), the alabaster pipe and the famous double pipes, accompanied on percussion instruments such as the tintinnabulum, tympanum and crotales, and later by stringed instruments like the lyre and kithara.",
            "Etruscan art was strongly connected to religion; the afterlife was of major importance in Etruscan art. The Etruscan musical instruments seen in frescoes and bas-reliefs are different types of pipes, such as the plagiaulos (the pipes of Pan or Syrinx), the alabaster pipe and the famous double pipes, accompanied on percussion instruments such as the tintinnabulum, tympanum and crotales, and later by stringed instruments like the lyre and kithara.",
            "Etruscan art was strongly connected to religion; the afterlife was of major importance in Etruscan art. The Etruscan musical instruments seen in frescoes and bas-reliefs are different types of pipes, such as the plagiaulos (the pipes of Pan or Syrinx), the alabaster pipe and the famous double pipes, accompanied on percussion instruments such as the tintinnabulum, tympanum and crotales, and later by stringed instruments like the lyre and kithara.",
            "Etruscan art was strongly connected to religion; the afterlife was of major importance in Etruscan art. The Etruscan musical instruments seen in frescoes and bas-reliefs are different types of pipes, such as the plagiaulos (the pipes of Pan or Syrinx), the alabaster pipe and the famous double pipes, accompanied on percussion instruments such as the tintinnabulum, tympanum and crotales, and later by stringed instruments like the lyre and kithara.",
            "Etruscan art was strongly connected to religion; the afterlife was of major importance in Etruscan art. The Etruscan musical instruments seen in frescoes and bas-reliefs are different types of pipes, such as the plagiaulos (the pipes of Pan or Syrinx), the alabaster pipe and the famous double pipes, accompanied on percussion instruments such as the tintinnabulum, tympanum and crotales, and later by stringed instruments like the lyre and kithara."
        ],
        "questions": [
            "What kind of art was the Etruscan known for?",
            "what shapes did they make?",
            "are there any pieces still around today?",
            "what type of music is the Etruscan civilization known for?",
            "what types of pipes?",
            "where did they play this music?",
            "is there anything noteworthy about their art and music in this article?",
            "what religion were they?"
        ]
    },
    {
        "title": "Doug Flutie",
        "contexts": [
            "Flutie played football for Boston College, the only Division I-A school to recruit him, from 1981 to 1984, and won the Heisman Trophy, Maxwell Award, and the Davey O'Brien National Quarterback Award in his senior year (1984). Flutie became the first quarterback to win the Heisman since Pat Sullivan in 1971. Flutie left school as the NCAA's all-time passing yardage leader with 10,579 yards and was a consensus All-American as a senior. He earned Player of the Year awards from UPI, Kodak, The Sporting News, and the Maxwell Football Club.",
            "Flutie gained national attention in 1984 when he led the Eagles to victory in a high-scoring, back-and-forth game against the Miami Hurricanes (led by QB Bernie Kosar). The game was nationally televised on CBS the day after Thanksgiving and thus had a huge audience. Miami staged a dramatic drive to take the lead, 45-41, in the closing minute of the game. Boston College then took possession at its own 22-yard line with 28 seconds to go. After two passes moved the ball another 30 yards, only 6 seconds remained. On the last play of the game, Flutie scrambled away from the defense and threw a \"Hail Mary pass\" that was caught in the end zone by Gerard Phelan, giving BC a 47-45 win.",
            "Flutie gained national attention in 1984 when he led the Eagles to victory in a high-scoring, back-and-forth game against the Miami Hurricanes (led by QB Bernie Kosar). The game was nationally televised on CBS the day after Thanksgiving and thus had a huge audience. Miami staged a dramatic drive to take the lead, 45-41, in the closing minute of the game. Boston College then took possession at its own 22-yard line with 28 seconds to go. After two passes moved the ball another 30 yards, only 6 seconds remained. On the last play of the game, Flutie scrambled away from the defense and threw a \"Hail Mary pass\" that was caught in the end zone by Gerard Phelan, giving BC a 47-45 win.",
            "Flutie became the first quarterback to win the Heisman since Pat Sullivan in 1971. Flutie left school as the NCAA's all-time passing yardage leader with 10,579 yards and was a consensus All-American as a senior. He earned Player of the Year awards from UPI, Kodak, The Sporting News, and the Maxwell Football Club. The quarterback coach for Boston College from 1981 - 1983 was Tom Coughlin.",
            "This idea essentially states that a winning sports team can increase the recognition value of a school enough to make it more attractive to potential applicants. In addition to his collegiate athletic achievement, Flutie maintained a distinguished academic record at Boston College. He was a candidate for a Rhodes Scholarship, for which he was named a finalist in 1984. Upon graduating, Flutie won a National Football Foundation post-graduate scholarship. In November 2008, Flutie was honored by Boston College with a statue of him throwing his famous \"Hail Mary\" pass outside of Alumni Stadium.",
            "This idea essentially states that a winning sports team can increase the recognition value of a school enough to make it more attractive to potential applicants. In addition to his collegiate athletic achievement, Flutie maintained a distinguished academic record at Boston College. He was a candidate for a Rhodes Scholarship, for which he was named a finalist in 1984. Upon graduating, Flutie won a National Football Foundation post-graduate scholarship. In November 2008, Flutie was honored by Boston College with a statue of him throwing his famous \"Hail Mary\" pass outside of Alumni Stadium.",
            "This idea essentially states that a winning sports team can increase the recognition value of a school enough to make it more attractive to potential applicants. In addition to his collegiate athletic achievement, Flutie maintained a distinguished academic record at Boston College. He was a candidate for a Rhodes Scholarship, for which he was named a finalist in 1984. Upon graduating, Flutie won a National Football Foundation post-graduate scholarship. In November 2008, Flutie was honored by Boston College with a statue of him throwing his famous \"Hail Mary\" pass outside of Alumni Stadium.",
            "Flutie became the first quarterback to win the Heisman since Pat Sullivan in 1971. Flutie left school as the NCAA's all-time passing yardage leader with 10,579 yards and was a consensus All-American as a senior. He earned Player of the Year awards from UPI, Kodak, The Sporting News, and the Maxwell Football Club. The quarterback coach for Boston College from 1981 - 1983 was Tom Coughlin.",
        ],
        "questions": [
            "What position did Doug Flutie play in college?",
            "What made his Hail Flutie pass so popular, was it record breaking?",
            "What are other plays he was known for?",
            "What are some of his college stats?",
            "Did he do anything else in college besides play football?",
            "Did he get visited by NFL scouts?",
            "Are there any other interesting aspects about this article?",
            "Are there any other records he set while playing in college?"
        ]
    },
    {
        "title": "Art Donovan",
        "contexts": [
            "He published an autobiography, Fatso, in 1987. He was noted as a jovial and humorous person during his playing career and capitalized on that with television and speaking appearances after retiring as a player. He owned and managed a country club near Baltimore. Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against.",
            "He published an autobiography, Fatso, in 1987. He was noted as a jovial and humorous person during his playing career and capitalized on that with television and speaking appearances after retiring as a player. He owned and managed a country club near Baltimore. Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against.",
            "He published an autobiography, Fatso, in 1987. He was noted as a jovial and humorous person during his playing career and capitalized on that with television and speaking appearances after retiring as a player. He owned and managed a country club near Baltimore. Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against.",
            "Donovan guest-starred in the Nickelodeon show The Adventures of Pete & Pete in the episode \"Space, Geeks, and Johnny Unitas\". He also appeared as a guest commentator at the WWF King of the Ring tournament in 1994. Donovan's appearance at the 1994 King of the Ring event would become infamous among wrestling fans for being seemingly uninformed about the product as well as generally befuddled behavior such as repeatedly asking how much certain wrestlers weighed.",
            "Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against. He relayed a story that he played without a helmet and in fact is shown on football cards without a helmet. Letterman wore Donovan's No. 70 Colts jersey in the famous Super Bowl XLI commercial with Oprah Winfrey and Jay Leno.",
            "He published an autobiography, Fatso, in 1987. He was noted as a jovial and humorous person during his playing career and capitalized on that with television and speaking appearances after retiring as a player. He owned and managed a country club near Baltimore. Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against.",
            "Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against. He relayed a story that he played without a helmet and in fact is shown on football cards without a helmet. Letterman wore Donovan's No. 70 Colts jersey in the famous Super Bowl XLI commercial with Oprah Winfrey and Jay Leno. Donovan guest-starred in the Nickelodeon show The Adventures of Pete & Pete in the episode \"Space, Geeks, and Johnny Unitas\". He also appeared as a guest commentator at the WWF King of the Ring tournament in 1994.",
            "Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against. He relayed a story that he played without a helmet and in fact is shown on football cards without a helmet. Letterman wore Donovan's No. 70 Colts jersey in the famous Super Bowl XLI commercial with Oprah Winfrey and Jay Leno. Donovan guest-starred in the Nickelodeon show The Adventures of Pete & Pete in the episode \"Space, Geeks, and Johnny Unitas\". He also appeared as a guest commentator at the WWF King of the Ring tournament in 1994.",
            "He published an autobiography, Fatso, in 1987. He was noted as a jovial and humorous person during his playing career and capitalized on that with television and speaking appearances after retiring as a player. He owned and managed a country club near Baltimore. Donovan also appeared ten times on the Late Show with David Letterman, telling humorous stories about his old playing days and about other \"old school\" footballers he played with and against."
        ],
        "questions": [
            "What did Art Donovan do when Donovan left the NFL?",
            "Was the autobiograpghy a hit",
            "What did he do after the autobigraphy was published",
            "Did he ever become a sports announcer?",
            "What kind of television and speaking appearances did he make",
            "When did he retire?",
            "What shows did he do besides Letterman and WWF",
            "Was that a one time guest appearance or multiple shows?",
            "Did he win any awards during this time"
        ]
    }
]

template_follow_up = """Write follow-up questions that have answers in the context in continuation of the previous questions.

{examples}

Title: {title}
Context: {context}
Question: {first_question}
Follow-up Question:"""


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_CHECKPOINT_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_CHECKPOINT_DIR,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    model = model.cuda()

    corpus = []
    with open(COLLECTION_JSONL) as f:
        for i, line in enumerate(tqdm(f)):
            data = json.loads(line.strip())
            assert i == data['id']
            corpus.append(data)

    corpus_title2ids = dict()
    for passage in tqdm(corpus):
        if passage['title'] not in corpus_title2ids:
            corpus_title2ids[passage['title']] = []
        corpus_title2ids[passage['title']].append(passage['id'])
    
    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stop, prompt_length):
            super().__init__()
            self.stop = stop
            self.prompt_length = prompt_length

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            return (self.stop == input_ids[:, self.prompt_length:]).any(dim=-1).all()

    def get_examples_first_turn(n_examples, shuffle=True):
        if shuffle:
            examples = random.sample(examples_first_turn, k=n_examples)
        else:
            examples = examples_first_turn[:n_examples]
        return "\n\n".join(examples)

    def get_examples_follow_up(n_examples, n_turns, shuffle=True):
        if shuffle:
            examples = random.sample(examples_follow_up, k=n_examples)
        else:
            examples = examples_follow_up[:n_examples]
        formatted_examples = []
        nl = "\n"
        for example in examples:
            max_turns = min(n_turns, len(example["questions"]))
            turns = []
            for i, (context, question) in enumerate(zip(example["contexts"][:max_turns], example["questions"][:max_turns])):
                prefix = "Question" if i == 0 else "Follow-up Question"
                turns.append(f"{prefix}: {example['questions'][i]}")
            formatted = f"Title: {example['title']}{nl}Context: {example['contexts'][max_turns-1]}{nl}{nl.join(turns)}"
            formatted_examples.append(formatted)
        
        return "\n\n".join(formatted_examples)
    
    new_line = tokenizer("\n").input_ids[-1]
    question_mark = tokenizer.convert_tokens_to_ids(["?"])[0]

    def is_degenerate(token_ids):
        if token_ids[-1] != new_line:
            return True
        token_ids = token_ids[:-1]
        
        if len(token_ids) > 0 and token_ids[-1] == question_mark:
            token_ids = token_ids[:-1]
        
        if len(token_ids) >= 3:
            if token_ids[-1] == token_ids[-2] == token_ids[-3]:
                return True
        if len(token_ids) >= 6:
            if token_ids[-1] == token_ids[-3] == token_ids[-5] and token_ids[-2] == token_ids[-4] == token_ids[-6]:
                return True
        
        return False
    
    passages_to_generate = [
        passage for passage in
        random.sample(corpus, k=N_PASSAGES)
        if len(passage['text'].split()) > 50
    ]

    generations = []
    n_examples_first_turn = 6
    n_examples_follow_up = 5
    max_length = 2048
    max_new_tokens_first_turn = 36
    max_new_tokens_follow_up = 24

    passage_switch_p = 0.2
    perturn_example = False
    shuffle_examples = False

    for passage in tqdm(passages_to_generate):
        turns = []
        passages = []
        
        dialog_length = random.choices([4, 5, 6, 7, 8, 9, 10, 11, 12], weights=[4, 5, 5, 8, 5, 4, 2, 1, 1], k=1)[0]

        n_tries = 0
        while len(turns) < dialog_length and n_tries < dialog_length * 1.5:
            if len(turns) == 0:
                examples = get_examples_first_turn(n_examples=n_examples_first_turn, shuffle=shuffle_examples)
                prompt = template_first_turn.format(examples=examples, title=passage['title'], context=passage['text'])
            else:
                sub = 0
                while sub < n_examples_follow_up:
                    examples = get_examples_follow_up(n_examples=n_examples_follow_up - sub, n_turns=len(turns) + 1 if perturn_example else 100, shuffle=shuffle_examples)
                    prompt = template_follow_up.format(examples=examples, title=passage['title'], context=passage['text'], first_question=turns[0])
                    for turn in turns[1:]:
                        prompt = prompt + f" {turn}\nFollow-up Question:"
                    if len(tokenizer(prompt).input_ids) < max_length - max_new_tokens_follow_up:
                        break
                    sub += 1
            
            input_ids = tokenizer(prompt, max_length=2048, truncation=True, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                stopping_criteria = StoppingCriteriaList(
                    [StoppingCriteriaSub(tokenizer("\n").input_ids[-1], input_ids.size(-1))]
                )
                outputs = model.generate(
                    input_ids,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.75,
                    min_new_tokens=1,
                    max_new_tokens=max_new_tokens_first_turn if len(turns) == 0 else max_new_tokens_follow_up,
                    stopping_criteria=stopping_criteria
                )
                
            generated_tokens = outputs[0][input_ids.size(-1):]
            turn = tokenizer.decode(generated_tokens).rstrip()
            
            if turn not in set(turns) and not is_degenerate(generated_tokens):
                turns.append(turn)
                passages.append(passage)
                if random.random() < passage_switch_p:
                    passage_id = random.choice(corpus_title2ids[passage['title']])
                    passage = corpus[passage_id]
            else:
                if turn in set(turns):
                    print(f"Found duplicate: {turn}")
                else:
                    print(f"Found degenerate: {turn}")
            
            n_tries += 1
            if len(turns) == 0 and n_tries >= 2:
                break
        
        generations.append({
            "passages": passages,
            "generation": turns
        })
    
    def write_generation_results(filename, generations, suffix):
        with open(filename + ".jsonl", "w") as f, open(filename + ".qrels.tsv", "w") as fqrels:
            for gen in generations:
                if len(gen["generation"]) != 7:
                    continue
                pid = gen['passage']['id']
                dialog_id = f"G_{pid}_{suffix}"
                for i in range(len(gen["generation"])):
                    qid = f"{dialog_id}_q#{i}"
                    f.write(json.dumps({"qid": qid, "input": gen["generation"][:i+1]}) + "\n")
                    fqrels.write(qid + "\t0\t" + str(pid) + "\t1\n")
                    
    def write_generation_results_passage_switch(filename, generations, suffix):
        with open(filename + ".jsonl", "w") as f, open(filename + ".qrels.tsv", "w") as fqrels:
            for gen in generations:
                if len(gen["generation"]) == 0:
                    continue
                assert len(gen["generation"]) == len(gen["passages"])
                pid = gen['passages'][0]['id']
                dialog_id = f"G_{pid}_{suffix}"
                for i in range(len(gen["generation"])):
                    pid = gen['passages'][i]['id']
                    qid = f"{dialog_id}_q#{i}"
                    f.write(json.dumps({"qid": qid, "input": gen["generation"][:i+1]}) + "\n")
                    fqrels.write(qid + "\t0\t" + str(pid) + "\t1\n")

    write_generation_results_passage_switch("ex6_ps_noshuffle_noperturn", generations, "0")
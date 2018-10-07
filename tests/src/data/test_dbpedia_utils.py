import pandas as pd
import pytest

from src.data.dbpedia_utils import (PHYSICISTS_IMPUTE_KEYS, PLACES_IMPUTE_KEYS,
                                    construct_resource_urls, impute_redirect_filenames)
from src.data.jsonl_utils import read_jsonl


@pytest.fixture(scope='module')
def read_redirects_cache():
    cache = pd.read_csv(
        'nobel_physics_prizes/data/raw/dbpedia-redirects.csv')
    redirect_urls = dict(zip(cache.url, cache.redirect_url))
    return redirect_urls


@pytest.fixture
def expected_albert_einstein_data():
    return {'abstract': 'Albert Einstein (/ˈaɪnstaɪn/; German: [ˈalbɛɐ̯t ˈaɪnʃtaɪn] ; 14 '
            'March 1879 – 18 April 1955) was a German-born theoretical '
            'physicist. He developed the general theory of relativity, one of '
            'the two pillars of modern physics (alongside quantum mechanics). '
            "Einstein's work is also known for its influence on the "
            'philosophy of science. Einstein is best known in popular culture '
            'for his mass–energy equivalence formula E = mc2 (which has been '
            'dubbed "the world\'s most famous equation"). He received the '
            '1921 Nobel Prize in Physics for his "services to theoretical '
            'physics", in particular his discovery of the law of the '
            'photoelectric effect, a pivotal step in the evolution of quantum '
            'theory. Near the beginning of his career, Einstein thought that '
            'Newtonian mechanics was no longer enough to reconcile the laws '
            'of classical mechanics with the laws of the electromagnetic '
            'field. This led to the development of his special theory of '
            'relativity. He realized, however, that the principle of '
            'relativity could also be extended to gravitational fields, and '
            'with his subsequent theory of gravitation in 1916, he published '
            'a paper on general relativity. He continued to deal with '
            'problems of statistical mechanics and quantum theory, which led '
            'to his explanations of particle theory and the motion of '
            'molecules. He also investigated the thermal properties of light '
            'which laid the foundation of the photon theory of light. In '
            '1917, Einstein applied the general theory of relativity to model '
            'the large-scale structure of the universe. He was visiting the '
            'United States when Adolf Hitler came to power in 1933 and, being '
            'Jewish, did not go back to Germany, where he had been a '
            'professor at the Berlin Academy of Sciences. He settled in the '
            'U.S., becoming an American citizen in 1940. On the eve of World '
            'War II, he endorsed a letter to President Franklin D. Roosevelt '
            'alerting him to the potential development of "extremely powerful '
            'bombs of a new type" and recommending that the U.S. begin '
            'similar research. This eventually led to what would become the '
            'Manhattan Project. Einstein supported defending the Allied '
            'forces, but largely denounced the idea of using the newly '
            'discovered nuclear fission as a weapon. Later, with the British '
            'philosopher Bertrand Russell, Einstein signed the '
            'Russell–Einstein Manifesto, which highlighted the danger of '
            'nuclear weapons. Einstein was affiliated with the Institute for '
            'Advanced Study in Princeton, New Jersey, until his death in '
            '1955. Einstein published more than 300 scientific papers along '
            'with over 150 non-scientific works. On 5 December 2014, '
            "universities and archives announced the release of Einstein's "
            "papers, comprising more than 30,000 unique documents. Einstein's "
            'intellectual achievements and originality have made the word '
            '"Einstein" synonymous with "genius".',
            'academicAdvisor': 'http://dbpedia.org/resource/Heinrich_Friedrich_Weber',
            'almaMater': 'http://dbpedia.org/resource/ETH_Zurich|http://dbpedia.org/resource/University_of_Zurich',
            'award': 'http://dbpedia.org/resource/Barnard_Medal_for_Meritorious_Service_to_Science|http://dbpedia.org/resource/Copley_Medal|http://dbpedia.org/resource/ForMemRS|http://dbpedia.org/resource/Gold_Medal_of_the_Royal_Astronomical_Society|http://dbpedia.org/resource/Matteucci_Medal|http://dbpedia.org/resource/Max_Planck_Medal|http://dbpedia.org/resource/Nobel_Prize_in_Physics|http://dbpedia.org/resource/Time_100:_The_Most_Important_People_of_the_Century',
            'birthDate': '1879-03-14',
            'birthPlace': 'http://dbpedia.org/resource/German_Empire|http://dbpedia.org/resource/Kingdom_of_Württemberg|http://dbpedia.org/resource/Ulm',
            'categories': 'http://dbpedia.org/resource/Category:1879_births|http://dbpedia.org/resource/Category:1955_deaths|http://dbpedia.org/resource/Category:20th-century_American_engineers|http://dbpedia.org/resource/Category:20th-century_American_writers|http://dbpedia.org/resource/Category:20th-century_German_writers|http://dbpedia.org/resource/Category:20th-century_Jews|http://dbpedia.org/resource/Category:20th-century_physicists|http://dbpedia.org/resource/Category:Albert_Einstein|http://dbpedia.org/resource/Category:American_agnostics|http://dbpedia.org/resource/Category:American_engineers|http://dbpedia.org/resource/Category:American_inventors|http://dbpedia.org/resource/Category:American_pacifists|http://dbpedia.org/resource/Category:American_people_of_German-Jewish_descent|http://dbpedia.org/resource/Category:American_philosophers|http://dbpedia.org/resource/Category:American_physicists|http://dbpedia.org/resource/Category:American_science_writers|http://dbpedia.org/resource/Category:American_socialists|http://dbpedia.org/resource/Category:American_Zionists|http://dbpedia.org/resource/Category:Anti-nationalists|http://dbpedia.org/resource/Category:Ashkenazi_Jews|http://dbpedia.org/resource/Category:Charles_University_in_Prague_faculty|http://dbpedia.org/resource/Category:Corresponding_Members_of_the_Russian_Academy_of_Sciences_(1917–25)|http://dbpedia.org/resource/Category:Cosmologists|http://dbpedia.org/resource/Category:Deaths_from_abdominal_aortic_aneurysm|http://dbpedia.org/resource/Category:Einstein_family|http://dbpedia.org/resource/Category:ETH_Zurich_alumni|http://dbpedia.org/resource/Category:ETH_Zurich_faculty|http://dbpedia.org/resource/Category:Foreign_Fellows_of_the_Indian_National_Science_Academy|http://dbpedia.org/resource/Category:Foreign_Members_of_the_Royal_Society|http://dbpedia.org/resource/Category:German_agnostics|http://dbpedia.org/resource/Category:German_emigrants_to_Switzerland|http://dbpedia.org/resource/Category:German_inventors|http://dbpedia.org/resource/Category:German_Nobel_laureates|http://dbpedia.org/resource/Category:German_physicists|http://dbpedia.org/resource/Category:German_socialists|http://dbpedia.org/resource/Category:Honorary_Members_of_the_USSR_Academy_of_Sciences|http://dbpedia.org/resource/Category:Institute_for_Advanced_Study_faculty|http://dbpedia.org/resource/Category:Jewish_agnostics|http://dbpedia.org/resource/Category:Jewish_American_scientists|http://dbpedia.org/resource/Category:Jewish_emigrants_from_Nazi_Germany_to_the_United_States|http://dbpedia.org/resource/Category:Jewish_engineers|http://dbpedia.org/resource/Category:Jewish_inventors|http://dbpedia.org/resource/Category:Jewish_philosophers|http://dbpedia.org/resource/Category:Jewish_physicists|http://dbpedia.org/resource/Category:Jewish_socialists|http://dbpedia.org/resource/Category:Leiden_University_faculty|http://dbpedia.org/resource/Category:Members_of_the_American_Philosophical_Society|http://dbpedia.org/resource/Category:Members_of_the_Bavarian_Academy_of_Sciences|http://dbpedia.org/resource/Category:Members_of_the_Lincean_Academy|http://dbpedia.org/resource/Category:Members_of_the_Royal_Netherlands_Academy_of_Arts_and_Sciences|http://dbpedia.org/resource/Category:Nobel_laureates_in_Physics|http://dbpedia.org/resource/Category:Patent_examiners|http://dbpedia.org/resource/Category:People_from_Berlin|http://dbpedia.org/resource/Category:People_from_Bern|http://dbpedia.org/resource/Category:People_from_Munich|http://dbpedia.org/resource/Category:People_from_Princeton,_New_Jersey|http://dbpedia.org/resource/Category:People_from_Ulm|http://dbpedia.org/resource/Category:People_from_Zürich|http://dbpedia.org/resource/Category:People_who_lost_German_citizenship|http://dbpedia.org/resource/Category:People_with_acquired_American_citizenship|http://dbpedia.org/resource/Category:People_with_acquired_Austrian_citizenship|http://dbpedia.org/resource/Category:People_with_acquired_Swiss_citizenship|http://dbpedia.org/resource/Category:Philosophers_of_science|http://dbpedia.org/resource/Category:Recipients_of_the_Pour_le_Mérite_(civil_class)|http://dbpedia.org/resource/Category:Recipients_of_the_Pour_le_Mérite_for_Arts_and_Sciences|http://dbpedia.org/resource/Category:Relativity_theorists|http://dbpedia.org/resource/Category:Sigma_Xi|http://dbpedia.org/resource/Category:Stateless_people|http://dbpedia.org/resource/Category:Subjects_of_iconic_photographs|http://dbpedia.org/resource/Category:Swiss_agnostics|http://dbpedia.org/resource/Category:Swiss_emigrants_to_the_United_States|http://dbpedia.org/resource/Category:Swiss_Jews|http://dbpedia.org/resource/Category:Swiss_physicists|http://dbpedia.org/resource/Category:Theoretical_physicists|http://dbpedia.org/resource/Category:Winners_of_the_Max_Planck_Medal|http://dbpedia.org/resource/Category:World_federalists',
            'child': 'http://dbpedia.org/resource/Hans_Albert_Einstein|http://dbpedia.org/resource/Lieserl_Einstein',
            'citizenship': 'http://dbpedia.org/resource/Austro-Hungarian_Empire|http://dbpedia.org/resource/Free_State_of_Prussia|http://dbpedia.org/resource/Kingdom_of_Prussia|http://dbpedia.org/resource/Kingdom_of_Württemberg|http://dbpedia.org/resource/Statelessness|http://dbpedia.org/resource/Switzerland|http://dbpedia.org/resource/Weimar_Republic',
            'comment': 'Albert Einstein (/ˈaɪnstaɪn/; German: [ˈalbɛɐ̯t ˈaɪnʃtaɪn] ; 14 '
            'March 1879 – 18 April 1955) was a German-born theoretical '
            'physicist. He developed the general theory of relativity, one of '
            'the two pillars of modern physics (alongside quantum mechanics). '
            "Einstein's work is also known for its influence on the philosophy "
            'of science. Einstein is best known in popular culture for his '
            'mass–energy equivalence formula E = mc2 (which has been dubbed '
            '"the world\'s most famous equation"). He received the 1921 Nobel '
            'Prize in Physics for his "services to theoretical physics", in '
            'particular his discovery of the law of the photoelectric effect, '
            'a pivotal step in the evolution of quantum theory.',
            'deathDate': '1955-04-18',
            'deathPlace': 'http://dbpedia.org/resource/Princeton,_New_Jersey',
            'description': 'German-American physicist and founder of the theory of '
            'relativity',
            'doctoralAdvisor': 'http://dbpedia.org/resource/Alfred_Kleiner',
            'field': 'http://dbpedia.org/resource/Philosophy|http://dbpedia.org/resource/Physics',
            'fullName': 'Albert Einstein',
            'gender': 'male',
            'givenName': 'Albert',
            'influenced': 'http://dbpedia.org/resource/Benjamin_Lee_Whorf|http://dbpedia.org/resource/Boris_Podolsky|http://dbpedia.org/resource/David_Bohm|http://dbpedia.org/resource/Émile_Meyerson|http://dbpedia.org/resource/Ernst_G._Straus|http://dbpedia.org/resource/Gunnar_Nordström|http://dbpedia.org/resource/Hans_Reichenbach|http://dbpedia.org/resource/Jean_Gebser|http://dbpedia.org/resource/John_Harnad|http://dbpedia.org/resource/Julius_Sumner_Miller|http://dbpedia.org/resource/Karl_Popper|http://dbpedia.org/resource/Keith_Lewin|http://dbpedia.org/resource/Leó_Szilárd|http://dbpedia.org/resource/Léon_Brillouin|http://dbpedia.org/resource/Mujahid_Kamran|http://dbpedia.org/resource/Nathan_Rosen|http://dbpedia.org/resource/Neil_deGrasse_Tyson|http://dbpedia.org/resource/Rajiv_Satyal|http://dbpedia.org/resource/Riazuddin_(physicist)|http://dbpedia.org/resource/Satyendra_Nath_Bose|http://dbpedia.org/resource/Sean_M._Carroll|http://dbpedia.org/resource/Théophile_de_Donder',
            'influencedBy': 'http://dbpedia.org/resource/Baruch_Spinoza|http://dbpedia.org/resource/David_Hume|http://dbpedia.org/resource/Ernst_Mach|http://dbpedia.org/resource/Gunnar_Nordström|http://dbpedia.org/resource/Henry_George|http://dbpedia.org/resource/Johann_Heinrich_Pestalozzi|http://dbpedia.org/resource/Karl_Pearson|http://dbpedia.org/resource/Moritz_Schlick',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Albert_Einstein',
            'knownFor': 'http://dbpedia.org/resource/Bose–Einstein_condensate|http://dbpedia.org/resource/Bose–Einstein_statistics|http://dbpedia.org/resource/Brownian_motion|http://dbpedia.org/resource/Classical_unified_field_theories|http://dbpedia.org/resource/Cosmological_constant|http://dbpedia.org/resource/Einstein_field_equations|http://dbpedia.org/resource/EPR_paradox|http://dbpedia.org/resource/General_relativity|http://dbpedia.org/resource/Gravitational_wave|http://dbpedia.org/resource/Mass–energy_equivalence|http://dbpedia.org/resource/Photoelectric_effect|http://dbpedia.org/resource/Special_relativity',
            'name': 'Albert Einstein',
            'residence': 'http://dbpedia.org/resource/Switzerland',
            'resource': 'http://dbpedia.org/resource/Albert_Einstein',
            'signature': 'Albert Einstein signature 1934.svg',
            'source': 'http://dbpedia.org/data/Albert_Einstein.json',
            'spouse': 'http://dbpedia.org/resource/Elsa_Löwenthal|http://dbpedia.org/resource/Mileva_Marić',
            'surname': 'Einstein',
            'theorized': 'http://dbpedia.org/resource/Photon',
            'thesisUrl': 'http://e-collection.library.ethz.ch/eserv/eth:30378/eth-30378-01.pdf',
            'thesisYear': 1905,
            'thumbnail': 'http://en.wikipedia.org/wiki/Special:FilePath/Einstein_1921_by_F_Schmutzer_-_restoration.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Albert_Einstein?oldid=745147482',
            'wikiPageID': 736,
            'wikiPageRevisionID': 745147482,
            'workplaces': 'http://dbpedia.org/resource/Caltech|http://dbpedia.org/resource/ETH_Zurich|http://dbpedia.org/resource/German_Physical_Society|http://dbpedia.org/resource/Humboldt_University_of_Berlin|http://dbpedia.org/resource/Institute_for_Advanced_Study|http://dbpedia.org/resource/Kaiser_Wilhelm_Institute|http://dbpedia.org/resource/Karl-Ferdinands-Universität|http://dbpedia.org/resource/Leiden_University|http://dbpedia.org/resource/Prussian_Academy_of_Sciences|http://dbpedia.org/resource/Swiss_Patent_Office|http://dbpedia.org/resource/University_of_Bern|http://dbpedia.org/resource/University_of_Zurich'
            }


@pytest.fixture
def expected_imputed_albert_einstein_data():
    return {'abstract': 'Albert Einstein (/ˈaɪnstaɪn/; German: [ˈalbɛɐ̯t ˈaɪnʃtaɪn] ; 14 '
            'March 1879 – 18 April 1955) was a German-born theoretical '
            'physicist. He developed the general theory of relativity, one of '
            'the two pillars of modern physics (alongside quantum mechanics). '
            "Einstein's work is also known for its influence on the "
            'philosophy of science. Einstein is best known in popular culture '
            'for his mass–energy equivalence formula E = mc2 (which has been '
            'dubbed "the world\'s most famous equation"). He received the '
            '1921 Nobel Prize in Physics for his "services to theoretical '
            'physics", in particular his discovery of the law of the '
            'photoelectric effect, a pivotal step in the evolution of quantum '
            'theory. Near the beginning of his career, Einstein thought that '
            'Newtonian mechanics was no longer enough to reconcile the laws '
            'of classical mechanics with the laws of the electromagnetic '
            'field. This led to the development of his special theory of '
            'relativity. He realized, however, that the principle of '
            'relativity could also be extended to gravitational fields, and '
            'with his subsequent theory of gravitation in 1916, he published '
            'a paper on general relativity. He continued to deal with '
            'problems of statistical mechanics and quantum theory, which led '
            'to his explanations of particle theory and the motion of '
            'molecules. He also investigated the thermal properties of light '
            'which laid the foundation of the photon theory of light. In '
            '1917, Einstein applied the general theory of relativity to model '
            'the large-scale structure of the universe. He was visiting the '
            'United States when Adolf Hitler came to power in 1933 and, being '
            'Jewish, did not go back to Germany, where he had been a '
            'professor at the Berlin Academy of Sciences. He settled in the '
            'U.S., becoming an American citizen in 1940. On the eve of World '
            'War II, he endorsed a letter to President Franklin D. Roosevelt '
            'alerting him to the potential development of "extremely powerful '
            'bombs of a new type" and recommending that the U.S. begin '
            'similar research. This eventually led to what would become the '
            'Manhattan Project. Einstein supported defending the Allied '
            'forces, but largely denounced the idea of using the newly '
            'discovered nuclear fission as a weapon. Later, with the British '
            'philosopher Bertrand Russell, Einstein signed the '
            'Russell–Einstein Manifesto, which highlighted the danger of '
            'nuclear weapons. Einstein was affiliated with the Institute for '
            'Advanced Study in Princeton, New Jersey, until his death in '
            '1955. Einstein published more than 300 scientific papers along '
            'with over 150 non-scientific works. On 5 December 2014, '
            "universities and archives announced the release of Einstein's "
            "papers, comprising more than 30,000 unique documents. Einstein's "
            'intellectual achievements and originality have made the word '
            '"Einstein" synonymous with "genius".',
            'academicAdvisor': 'Heinrich Friedrich Weber',
            'almaMater': 'ETH Zurich|University of Zurich',
            'award': 'Barnard Medal for Meritorious Service to Science|Copley '
            'Medal|Fellow of the Royal Society|Gold Medal of the Royal '
            'Astronomical Society|Matteucci Medal|Max Planck Medal|Nobel Prize '
            'in Physics|Time 100: The Most Important People of the Century',
            'birthDate': '1879-03-14',
            'birthPlace': 'German Empire|Kingdom of Württemberg|Ulm',
            'categories': '1879 births|1955 deaths|20th-century American '
            'engineers|20th-century American writers|20th-century German '
            'writers|20th-century Jews|20th-century physicists|Albert '
            'Einstein|American agnostics|American engineers|American '
            'inventors|American pacifists|American people of German-Jewish '
            'descent|American philosophers|American physicists|American '
            'science writers|American socialists|American '
            'Zionists|Anti-nationalists|Ashkenazi Jews|Charles University '
            'in Prague faculty|Corresponding Members of the Russian Academy '
            'of Sciences (1917–25)|Cosmologists|Deaths from abdominal '
            'aortic aneurysm|Einstein family|ETH Zurich alumni|ETH Zurich '
            'faculty|Foreign Fellows of the Indian National Science '
            'Academy|Foreign Members of the Royal Society|German '
            'agnostics|German emigrants to Switzerland|German '
            'inventors|German Nobel laureates|German physicists|German '
            'socialists|Honorary Members of the USSR Academy of '
            'Sciences|Institute for Advanced Study faculty|Jewish '
            'agnostics|Jewish American scientists|Jewish emigrants from '
            'Nazi Germany to the United States|Jewish engineers|Jewish '
            'inventors|Jewish philosophers|Jewish physicists|Jewish '
            'socialists|Leiden University faculty|Members of the American '
            'Philosophical Society|Members of the Bavarian Academy of '
            'Sciences|Members of the Lincean Academy|Members of the Royal '
            'Netherlands Academy of Arts and Sciences|Nobel laureates in '
            'Physics|Patent examiners|People from Berlin|People from '
            'Bern|People from Munich|People from Princeton, New '
            'Jersey|People from Ulm|People from Zürich|People who lost '
            'German citizenship|People with acquired American '
            'citizenship|People with acquired Austrian citizenship|People '
            'with acquired Swiss citizenship|Philosophers of '
            'science|Recipients of the Pour le Mérite (civil '
            'class)|Recipients of the Pour le Mérite for Arts and '
            'Sciences|Relativity theorists|Sigma Xi|Stateless '
            'people|Subjects of iconic photographs|Swiss agnostics|Swiss '
            'emigrants to the United States|Swiss Jews|Swiss '
            'physicists|Theoretical physicists|Winners of the Max Planck '
            'Medal|World federalists',
            'child': 'Einstein family|Hans Albert Einstein',
            'citizenship': 'Austria-Hungary|Free State of Prussia|Kingdom of '
            'Prussia|Kingdom of '
            'Württemberg|Statelessness|Switzerland|Weimar Republic',
            'comment': 'Albert Einstein (/ˈaɪnstaɪn/; German: [ˈalbɛɐ̯t ˈaɪnʃtaɪn] ; 14 '
            'March 1879 – 18 April 1955) was a German-born theoretical '
            'physicist. He developed the general theory of relativity, one of '
            'the two pillars of modern physics (alongside quantum mechanics). '
            "Einstein's work is also known for its influence on the philosophy "
            'of science. Einstein is best known in popular culture for his '
            'mass–energy equivalence formula E = mc2 (which has been dubbed '
            '"the world\'s most famous equation"). He received the 1921 Nobel '
            'Prize in Physics for his "services to theoretical physics", in '
            'particular his discovery of the law of the photoelectric effect, '
            'a pivotal step in the evolution of quantum theory.',
            'deathDate': '1955-04-18',
            'deathPlace': 'Princeton, New Jersey',
            'description': 'German-American physicist and founder of the theory of '
            'relativity',
            'doctoralAdvisor': 'Alfred Kleiner',
            'field': 'Philosophy|Physics',
            'fullName': 'Albert Einstein',
            'gender': 'male',
            'givenName': 'Albert',
            'influenced': 'Benjamin Lee Whorf|Boris Podolsky|David Bohm|Émile '
            'Meyerson|Ernst G. Straus|Gunnar Nordström|Hans '
            'Reichenbach|Jean Gebser|John Harnad|Julius Sumner Miller|Karl '
            'Popper|Keith Lewin|Leo Szilard|Léon Brillouin|Mujahid '
            'Kamran|Nathan Rosen|Neil deGrasse Tyson|Rajiv Satyal|Riazuddin '
            '(physicist)|Satyendra Nath Bose|Sean M. Carroll|Théophile de '
            'Donder',
            'influencedBy': 'Baruch Spinoza|David Hume|Ernst Mach|Gunnar Nordström|Henry '
            'George|Johann Heinrich Pestalozzi|Karl Pearson|Moritz '
            'Schlick',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Albert_Einstein',
            'knownFor': 'Bose–Einstein condensate|Bose–Einstein statistics|Brownian '
            'motion|Classical unified field theories|Cosmological '
            'constant|Einstein field equations|EPR paradox|General '
            'relativity|Gravitational wave|Mass–energy '
            'equivalence|Photoelectric effect|Special relativity',
            'name': 'Albert Einstein',
            'residence': 'Switzerland',
            'resource': 'http://dbpedia.org/resource/Albert_Einstein',
            'signature': 'Albert Einstein signature 1934.svg',
            'source': 'http://dbpedia.org/data/Albert_Einstein.json',
            'spouse': 'Elsa Einstein|Mileva Marić',
            'surname': 'Einstein',
            'theorized': 'Photon',
            'thesisUrl': 'http://e-collection.library.ethz.ch/eserv/eth:30378/eth-30378-01.pdf',
            'thesisYear': 1905,
            'thumbnail': 'http://en.wikipedia.org/wiki/Special:FilePath/Einstein_1921_by_F_Schmutzer_-_restoration.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Albert_Einstein?oldid=745147482',
            'wikiPageID': 736,
            'wikiPageRevisionID': 745147482,
            'workplaces': 'California Institute of Technology|Deutsche Physikalische '
            'Gesellschaft|ETH Zurich|Humboldt University of '
            'Berlin|Institute for Advanced Study|Kaiser Wilhelm '
            'Society|Karl-Ferdinands-Universität|Leiden University|Prussian '
            'Academy of Sciences|Swiss Federal Institute of Intellectual '
            'Property|University of Bern|University of Zurich'
            }


@pytest.fixture
def expected_marie_curie_data():
    return {'abstract': 'Marie Skłodowska Curie (/ˈkjʊri, kjʊˈriː/; French: [kyʁi]; '
            'Polish: [kʲiˈri]; 7 November 1867 – 4 July 1934), born Maria '
            'Salomea Skłodowska [ˈmarja salɔˈmɛa skwɔˈdɔfska], was a Polish '
            'and naturalized-French physicist and chemist who conducted '
            'pioneering research on radioactivity. She was the first woman to '
            'win a Nobel Prize, the first person and only woman to win twice, '
            'the only person to win twice in multiple sciences, and was part '
            'of the Curie family legacy of five Nobel Prizes. She was also '
            'the first woman to become a professor at the University of '
            'Paris, and in 1995 became the first woman to be entombed on her '
            'own merits in the Panthéon in Paris. She was born in Warsaw, in '
            'what was then the Kingdom of Poland, part of the Russian Empire. '
            "She studied at Warsaw's clandestine Floating University and "
            'began her practical scientific training in Warsaw. In 1891, aged '
            '24, she followed her older sister Bronisława to study in Paris, '
            'where she earned her higher degrees and conducted her subsequent '
            'scientific work. She shared the 1903 Nobel Prize in Physics with '
            'her husband Pierre Curie and with physicist Henri Becquerel. She '
            'won the 1911 Nobel Prize in Chemistry. Her achievements included '
            'the development of the theory of radioactivity (a term that she '
            'coined), techniques for isolating radioactive isotopes, and the '
            'discovery of two elements, polonium and radium. Under her '
            "direction, the world's first studies were conducted into the "
            'treatment of neoplasms, using radioactive isotopes. She founded '
            'the Curie Institutes in Paris and in Warsaw, which remain major '
            'centres of medical research today. During World War I, she '
            'established the first military field radiological centres. While '
            'a French citizen, Marie Skłodowska Curie (she used both '
            'surnames) never lost her sense of Polish identity. She taught '
            'her daughters the Polish language and took them on visits to '
            'Poland. She named the first chemical element that she '
            'discovered\u200d—\u200cpolonium, which she isolated in '
            '1898\u200d—\u200cafter her native country. Curie died in 1934, '
            'aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, '
            'due to aplastic anemia brought on by exposure to radiation while '
            'carrying test tubes of radium in her pockets during research, '
            'and in the course of her service in World War I mobile X-ray '
            'units that she had set up.',
            'almaMater': 'http://dbpedia.org/resource/ESPCI',
            'award': 'http://dbpedia.org/resource/Albert_Medal_(Royal_Society_of_Arts)|http://dbpedia.org/resource/Davy_Medal|http://dbpedia.org/resource/Elliott_Cresson_Medal|http://dbpedia.org/resource/Matteucci_Medal|http://dbpedia.org/resource/Nobel_Prize_in_Chemistry|http://dbpedia.org/resource/Nobel_Prize_in_Physics|http://dbpedia.org/resource/Willard_Gibbs_Award',
            'birthDate': '1867-11-07',
            'birthName': 'Maria Salomea Skłodowska',
            'birthPlace': 'http://dbpedia.org/resource/Congress_Poland|http://dbpedia.org/resource/Russian_Empire|http://dbpedia.org/resource/Warsaw',
            'categories': "http://dbpedia.org/resource/Category:1867_births|http://dbpedia.org/resource/Category:1934_deaths|http://dbpedia.org/resource/Category:19th-century_physicists|http://dbpedia.org/resource/Category:19th-century_women_scientists|http://dbpedia.org/resource/Category:20th-century_physicists|http://dbpedia.org/resource/Category:20th-century_women_scientists|http://dbpedia.org/resource/Category:Burials_at_the_Panthéon,_Paris|http://dbpedia.org/resource/Category:Congress_Poland_emigrants_to_France|http://dbpedia.org/resource/Category:Corresponding_Members_of_the_Russian_Academy_of_Sciences_(1917–25)|http://dbpedia.org/resource/Category:Corresponding_Members_of_the_St_Petersburg_Academy_of_Sciences|http://dbpedia.org/resource/Category:Corresponding_Members_of_the_USSR_Academy_of_Sciences|http://dbpedia.org/resource/Category:Curie_family|http://dbpedia.org/resource/Category:Deaths_from_anemia|http://dbpedia.org/resource/Category:Deaths_from_cancer_in_France|http://dbpedia.org/resource/Category:Discoverers_of_chemical_elements|http://dbpedia.org/resource/Category:Experimental_physicists|http://dbpedia.org/resource/Category:Former_Roman_Catholics|http://dbpedia.org/resource/Category:French_chemists|http://dbpedia.org/resource/Category:French_Nobel_laureates|http://dbpedia.org/resource/Category:French_people_of_Polish_descent|http://dbpedia.org/resource/Category:French_physicists|http://dbpedia.org/resource/Category:Governesses|http://dbpedia.org/resource/Category:Honorary_Members_of_the_USSR_Academy_of_Sciences|http://dbpedia.org/resource/Category:Légion_d'honneur_refusals|http://dbpedia.org/resource/Category:Marie_Curie|http://dbpedia.org/resource/Category:Members_of_the_Lwów_Scientific_Society|http://dbpedia.org/resource/Category:Naturalized_citizens_of_France|http://dbpedia.org/resource/Category:Nobel_laureates_in_Chemistry|http://dbpedia.org/resource/Category:Nobel_laureates_in_Physics|http://dbpedia.org/resource/Category:Nobel_laureates_with_multiple_Nobel_awards|http://dbpedia.org/resource/Category:Nuclear_chemists|http://dbpedia.org/resource/Category:People_from_Warsaw|http://dbpedia.org/resource/Category:People_from_Warsaw_Governorate|http://dbpedia.org/resource/Category:Polish_agnostics|http://dbpedia.org/resource/Category:Polish_chemists|http://dbpedia.org/resource/Category:Polish_Nobel_laureates|http://dbpedia.org/resource/Category:Polish_physicists|http://dbpedia.org/resource/Category:Radioactivity|http://dbpedia.org/resource/Category:University_of_Paris_alumni|http://dbpedia.org/resource/Category:University_of_Paris_faculty|http://dbpedia.org/resource/Category:Women_chemists|http://dbpedia.org/resource/Category:Women_Nobel_laureates|http://dbpedia.org/resource/Category:Women_physicists",
            'child': 'http://dbpedia.org/resource/Ève_Curie|http://dbpedia.org/resource/Irène_Joliot-Curie',
            'citizenship': 'Poland \n* France',
            'comment': 'Marie Skłodowska Curie (/ˈkjʊri, kjʊˈriː/; French: [kyʁi]; '
            'Polish: [kʲiˈri]; 7 November 1867 – 4 July 1934), born Maria '
            'Salomea Skłodowska [ˈmarja salɔˈmɛa skwɔˈdɔfska], was a Polish '
            'and naturalized-French physicist and chemist who conducted '
            'pioneering research on radioactivity. She was the first woman to '
            'win a Nobel Prize, the first person and only woman to win twice, '
            'the only person to win twice in multiple sciences, and was part '
            'of the Curie family legacy of five Nobel Prizes. She was also the '
            'first woman to become a professor at the University of Paris, and '
            'in 1995 became the first woman to be entombed on her own merits '
            'in the Panthéon in Paris.',
            'deathDate': '1934-07-04',
            'deathPlace': 'http://dbpedia.org/resource/Passy,_Haute-Savoie|http://dbpedia.org/resource/Sancellemoz',
            'description': 'French-Polish physicist and chemist',
            'doctoralAdvisor': 'http://dbpedia.org/resource/Gabriel_Lippmann',
            'doctoralStudent': 'http://dbpedia.org/resource/André-Louis_Debierne|http://dbpedia.org/resource/Émile_Henriot_(chemist)|http://dbpedia.org/resource/Marguerite_Perey|http://dbpedia.org/resource/Óscar_Moreno',
            'field': 'http://dbpedia.org/resource/Chemistry|http://dbpedia.org/resource/Physics',
            'fullName': 'Marie Curie',
            'gender': 'female',
            'givenName': 'Maria',
            'influenced': 'http://dbpedia.org/resource/Robert_Abbe',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Marie_Curie',
            'knownFor': 'http://dbpedia.org/resource/Polonium|http://dbpedia.org/resource/Radioactivity|http://dbpedia.org/resource/Radium',
            'name': 'Marie Curie|Marie Skłodowska Curie',
            'residence': 'Poland, France',
            'resource': 'http://dbpedia.org/resource/Marie_Curie',
            'signature': 'Marie Curie Skłodowska Signature Polish.svg',
            'source': 'http://dbpedia.org/data/Marie_Curie.json',
            'spouse': 'http://dbpedia.org/resource/Pierre_Curie',
            'surname': 'Curie',
            'thesisTitle': 'Recherches sur les substances radioactives',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Marie_Curie_c1920.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Marie_Curie?oldid=745192875',
            'wikiPageID': 20408,
            'wikiPageRevisionID': 745192875,
            'workplaces': 'University of Paris\n'
            '* École Normale Supérieure\n'
            '* Curie Institute (Paris) \n'
            '* French Academy of Medicine\n'
            '* International Committee on Intellectual Cooperation'}


@pytest.fixture
def expected_imputed_marie_curie_data():
    return {'abstract': 'Marie Skłodowska Curie (/ˈkjʊri, kjʊˈriː/; French: [kyʁi]; '
            'Polish: [kʲiˈri]; 7 November 1867 – 4 July 1934), born Maria '
            'Salomea Skłodowska [ˈmarja salɔˈmɛa skwɔˈdɔfska], was a Polish '
            'and naturalized-French physicist and chemist who conducted '
            'pioneering research on radioactivity. She was the first woman to '
            'win a Nobel Prize, the first person and only woman to win twice, '
            'the only person to win twice in multiple sciences, and was part '
            'of the Curie family legacy of five Nobel Prizes. She was also '
            'the first woman to become a professor at the University of '
            'Paris, and in 1995 became the first woman to be entombed on her '
            'own merits in the Panthéon in Paris. She was born in Warsaw, in '
            'what was then the Kingdom of Poland, part of the Russian Empire. '
            "She studied at Warsaw's clandestine Floating University and "
            'began her practical scientific training in Warsaw. In 1891, aged '
            '24, she followed her older sister Bronisława to study in Paris, '
            'where she earned her higher degrees and conducted her subsequent '
            'scientific work. She shared the 1903 Nobel Prize in Physics with '
            'her husband Pierre Curie and with physicist Henri Becquerel. She '
            'won the 1911 Nobel Prize in Chemistry. Her achievements included '
            'the development of the theory of radioactivity (a term that she '
            'coined), techniques for isolating radioactive isotopes, and the '
            'discovery of two elements, polonium and radium. Under her '
            "direction, the world's first studies were conducted into the "
            'treatment of neoplasms, using radioactive isotopes. She founded '
            'the Curie Institutes in Paris and in Warsaw, which remain major '
            'centres of medical research today. During World War I, she '
            'established the first military field radiological centres. While '
            'a French citizen, Marie Skłodowska Curie (she used both '
            'surnames) never lost her sense of Polish identity. She taught '
            'her daughters the Polish language and took them on visits to '
            'Poland. She named the first chemical element that she '
            'discovered\u200d—\u200cpolonium, which she isolated in '
            '1898\u200d—\u200cafter her native country. Curie died in 1934, '
            'aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, '
            'due to aplastic anemia brought on by exposure to radiation while '
            'carrying test tubes of radium in her pockets during research, '
            'and in the course of her service in World War I mobile X-ray '
            'units that she had set up.',
            'almaMater': 'ESPCI ParisTech',
            'award': 'Albert Medal (Royal Society of Arts)|Davy Medal|Elliott Cresson '
            'Medal|Matteucci Medal|Nobel Prize in Chemistry|Nobel Prize in '
            'Physics|Willard Gibbs Award',
            'birthDate': '1867-11-07',
            'birthName': 'Maria Salomea Skłodowska',
            'birthPlace': 'Congress Poland|Russian Empire|Warsaw',
            'categories': '1867 births|1934 deaths|19th-century physicists|19th-century '
            'women scientists|20th-century physicists|20th-century women '
            'scientists|Burials at the Panthéon, Paris|Congress Poland '
            'emigrants to France|Corresponding Members of the Russian '
            'Academy of Sciences (1917–25)|Corresponding Members of the St '
            'Petersburg Academy of Sciences|Corresponding Members of the '
            'USSR Academy of Sciences|Curie family|Deaths from '
            'anemia|Deaths from cancer in France|Discoverers of chemical '
            'elements|Experimental physicists|Former Roman Catholics|French '
            'chemists|French Nobel laureates|French people of Polish '
            'descent|French physicists|Governesses|Honorary Members of the '
            "USSR Academy of Sciences|Légion d'honneur refusals|Marie "
            'Curie|Members of the Lwów Scientific Society|Naturalized '
            'citizens of France|Nobel laureates in Chemistry|Nobel '
            'laureates in Physics|Nobel laureates with multiple Nobel '
            'awards|Nuclear chemists|People from Warsaw|People from Warsaw '
            'Governorate|Polish agnostics|Polish chemists|Polish Nobel '
            'laureates|Polish physicists|Radioactivity|University of Paris '
            'alumni|University of Paris faculty|Women chemists|Women Nobel '
            'laureates|Women physicists',
            'child': 'Ève Curie|Irène Joliot-Curie',
            'citizenship': 'France|Poland',
            'comment': 'Marie Skłodowska Curie (/ˈkjʊri, kjʊˈriː/; French: [kyʁi]; '
            'Polish: [kʲiˈri]; 7 November 1867 – 4 July 1934), born Maria '
            'Salomea Skłodowska [ˈmarja salɔˈmɛa skwɔˈdɔfska], was a Polish '
            'and naturalized-French physicist and chemist who conducted '
            'pioneering research on radioactivity. She was the first woman to '
            'win a Nobel Prize, the first person and only woman to win twice, '
            'the only person to win twice in multiple sciences, and was part '
            'of the Curie family legacy of five Nobel Prizes. She was also the '
            'first woman to become a professor at the University of Paris, and '
            'in 1995 became the first woman to be entombed on her own merits '
            'in the Panthéon in Paris.',
            'deathDate': '1934-07-04',
            'deathPlace': 'Passy, Haute-Savoie|Sancellemoz',
            'description': 'French-Polish physicist and chemist',
            'doctoralAdvisor': 'Gabriel Lippmann',
            'doctoralStudent': 'André-Louis Debierne|Émile Henriot (chemist)|Marguerite '
            'Perey|Óscar Moreno',
            'field': 'Chemistry|Physics',
            'fullName': 'Marie Curie',
            'gender': 'female',
            'givenName': 'Maria',
            'influenced': 'Robert Abbe',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Marie_Curie',
            'knownFor': 'Polonium|Radioactive decay|Radium',
            'name': 'Marie Curie|Marie Skłodowska Curie',
            'residence': 'Poland, France',
            'resource': 'http://dbpedia.org/resource/Marie_Curie',
            'signature': 'Marie Curie Skłodowska Signature Polish.svg',
            'source': 'http://dbpedia.org/data/Marie_Curie.json',
            'spouse': 'Pierre Curie',
            'surname': 'Curie',
            'thesisTitle': 'Recherches sur les substances radioactives',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Marie_Curie_c1920.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Marie_Curie?oldid=745192875',
            'wikiPageID': 20408,
            'wikiPageRevisionID': 745192875,
            'workplaces': 'Curie Institute (Paris)|École Normale Supérieure|French '
            'Academy of Medicine|International Committee on Intellectual '
            'Cooperation|University of Paris'}


@pytest.fixture
def expected_max_born_data():
    return {'abstract': 'Max Born (German: [bɔɐ̯n]; 11 December 1882 – 5 January 1970) '
            'was a German physicist and mathematician who was instrumental in '
            'the development of quantum mechanics. He also made contributions '
            'to solid-state physics and optics and supervised the work of a '
            'number of notable physicists in the 1920s and 1930s. Born won '
            'the 1954 Nobel Prize in Physics for his "fundamental research in '
            'Quantum Mechanics, especially in the statistical interpretation '
            'of the wave function". Born in 1882 in Breslau, then in Germany, '
            'now in Poland and known as Wrocław, Born entered the University '
            'of Göttingen in 1904, where he found the three renowned '
            'mathematicians, Felix Klein, David Hilbert and Hermann '
            'Minkowski. He wrote his Ph.D. thesis on the subject of '
            '"Stability of Elastica in a Plane and Space", winning the '
            "University's Philosophy Faculty Prize. In 1905, he began "
            'researching special relativity with Minkowski, and subsequently '
            'wrote his habilitation thesis on the Thomson model of the atom. '
            'A chance meeting with Fritz Haber in Berlin in 1918 led to '
            'discussion of the manner in which an ionic compound is formed '
            'when a metal reacts with a halogen, which is today known as the '
            'Born–Haber cycle. In the First World War, after originally being '
            'placed as a radio operator, he was moved to research duties '
            'regarding sound ranging due to his specialist knowledge. In '
            '1921, Born returned to Göttingen, arranging another chair for '
            'his long-time friend and colleague James Franck. Under Born, '
            "Göttingen became one of the world's foremost centres for "
            'physics. In 1925, Born and Werner Heisenberg formulated the '
            'matrix mechanics representation of quantum mechanics. The '
            'following year, he formulated the now-standard interpretation of '
            'the probability density function for ψ*ψ in the Schrödinger '
            'equation, for which he was awarded the Nobel Prize in 1954. His '
            'influence extended far beyond his own research. Max Delbrück, '
            'Siegfried Flügge, Friedrich Hund, Pascual Jordan, Maria '
            'Goeppert-Mayer, Lothar Wolfgang Nordheim, Robert Oppenheimer, '
            'and Victor Weisskopf all received their Ph.D. degrees under Born '
            'at Göttingen, and his assistants included Enrico Fermi, Werner '
            'Heisenberg, Gerhard Herzberg, Friedrich Hund, Pascual Jordan, '
            'Wolfgang Pauli, Léon Rosenfeld, Edward Teller, and Eugene '
            'Wigner. In January 1933, the Nazi Party came to power in '
            'Germany, and Born, who was Jewish, was suspended. He emigrated '
            "to Britain, where he took a job at St John's College, Cambridge, "
            'and wrote a popular science book, The Restless Universe, as well '
            'as Atomic Physics, which soon became a standard textbook. In '
            'October 1936, he became the Tait Professor of Natural Philosophy '
            'at the University of Edinburgh, where, working with German-born '
            'assistants E. Walter Kellermann and Klaus Fuchs, he continued '
            'his research into physics. Max Born became a naturalised British '
            'subject on 31 August 1939, one day before World War II broke out '
            'in Europe. He remained at Edinburgh until 1952. He retired to '
            'Bad Pyrmont, in West Germany, and died in a hospital in '
            'Göttingen on 5 January 1970.',
            'academicAdvisor': 'http://dbpedia.org/resource/J._J._Thomson|http://dbpedia.org/resource/Joseph_Larmor|http://dbpedia.org/resource/Karl_Schwarzschild|http://dbpedia.org/resource/Woldemar_Voigt',
            'almaMater': 'http://dbpedia.org/resource/University_of_Göttingen',
            'award': 'http://dbpedia.org/resource/Fellow_of_the_Royal_Society|http://dbpedia.org/resource/Hughes_Medal|http://dbpedia.org/resource/Max_Planck_Medal|http://dbpedia.org/resource/Nobel_Prize_in_Physics',
            'birthDate': '1882-12-11',
            'birthPlace': 'http://dbpedia.org/resource/German_Empire|http://dbpedia.org/resource/Wrocław',
            'categories': 'http://dbpedia.org/resource/Category:1882_births|http://dbpedia.org/resource/Category:1970_deaths|http://dbpedia.org/resource/Category:Academics_of_the_University_of_Cambridge|http://dbpedia.org/resource/Category:Academics_of_the_University_of_Edinburgh|http://dbpedia.org/resource/Category:Alumni_of_Gonville_and_Caius_College,_Cambridge|http://dbpedia.org/resource/Category:British_Jews|http://dbpedia.org/resource/Category:British_Lutherans|http://dbpedia.org/resource/Category:British_people_of_German-Jewish_descent|http://dbpedia.org/resource/Category:British_physicists|http://dbpedia.org/resource/Category:Deists|http://dbpedia.org/resource/Category:Fellows_of_the_Royal_Society|http://dbpedia.org/resource/Category:Fellows_of_the_Royal_Society_of_Edinburgh|http://dbpedia.org/resource/Category:Foreign_Members_of_the_USSR_Academy_of_Sciences|http://dbpedia.org/resource/Category:German_emigrants_to_Scotland|http://dbpedia.org/resource/Category:German_Jews|http://dbpedia.org/resource/Category:German_Lutherans|http://dbpedia.org/resource/Category:German_Nobel_laureates|http://dbpedia.org/resource/Category:German_physicists|http://dbpedia.org/resource/Category:Goethe_University_Frankfurt_faculty|http://dbpedia.org/resource/Category:Grand_Crosses_with_Star_and_Sash_of_the_Order_of_Merit_of_the_Federal_Republic_of_Germany|http://dbpedia.org/resource/Category:Heidelberg_University_alumni|http://dbpedia.org/resource/Category:Honorary_Members_of_the_USSR_Academy_of_Sciences|http://dbpedia.org/resource/Category:Humboldt_University_of_Berlin_faculty|http://dbpedia.org/resource/Category:Jewish_physicists|http://dbpedia.org/resource/Category:Jews_who_immigrated_to_the_United_Kingdom_to_escape_Nazism|http://dbpedia.org/resource/Category:Members_of_the_Prussian_Academy_of_Sciences|http://dbpedia.org/resource/Category:Nobel_laureates_in_Physics|http://dbpedia.org/resource/Category:Optical_physicists|http://dbpedia.org/resource/Category:People_associated_with_the_University_of_Zurich|http://dbpedia.org/resource/Category:People_from_the_Province_of_Silesia|http://dbpedia.org/resource/Category:People_from_Wrocław|http://dbpedia.org/resource/Category:People_who_emigrated_to_escape_Nazism|http://dbpedia.org/resource/Category:Quantum_physicists|http://dbpedia.org/resource/Category:Scientists_from_Frankfurt|http://dbpedia.org/resource/Category:Silesian_Jews|http://dbpedia.org/resource/Category:Theoretical_physicists|http://dbpedia.org/resource/Category:University_of_Breslau_alumni|http://dbpedia.org/resource/Category:University_of_Göttingen_alumni|http://dbpedia.org/resource/Category:University_of_Göttingen_faculty|http://dbpedia.org/resource/Category:Winners_of_the_Max_Planck_Medal',
            'child': 'eldest  was mother of Olivia Newton-John',
            'citizenship': 'British|German',
            'comment': 'Max Born (German: [bɔɐ̯n]; 11 December 1882 – 5 January 1970) was '
            'a German physicist and mathematician who was instrumental in the '
            'development of quantum mechanics. He also made contributions to '
            'solid-state physics and optics and supervised the work of a '
            'number of notable physicists in the 1920s and 1930s. Born won the '
            '1954 Nobel Prize in Physics for his "fundamental research in '
            'Quantum Mechanics, especially in the statistical interpretation '
            'of the wave function".',
            'deathDate': '1970-01-05',
            'deathPlace': 'http://dbpedia.org/resource/Göttingen|http://dbpedia.org/resource/West_Germany',
            'description': 'physicist',
            'doctoralAdvisor': 'http://dbpedia.org/resource/Carl_Runge',
            'doctoralStudent': 'http://dbpedia.org/resource/Bertha_Swirles|http://dbpedia.org/resource/Cheng_Kaijia|http://dbpedia.org/resource/Edgar_Krahn|http://dbpedia.org/resource/Friedrich_Hund|http://dbpedia.org/resource/Herbert_S._Green|http://dbpedia.org/resource/J._Robert_Oppenheimer|http://dbpedia.org/resource/Lothar_Wolfgang_Nordheim|http://dbpedia.org/resource/Maria_Goeppert-Mayer|http://dbpedia.org/resource/Maurice_Pryce|http://dbpedia.org/resource/Max_Delbrück|http://dbpedia.org/resource/Pascual_Jordan|http://dbpedia.org/resource/Peng_Huanwu|http://dbpedia.org/resource/Siegfried_Flügge|http://dbpedia.org/resource/Victor_Frederick_Weisskopf|http://dbpedia.org/resource/Walter_Elsasser',
            'field': 'http://dbpedia.org/resource/Physics',
            'fullName': 'Max Born',
            'gender': 'male',
            'givenName': 'Max',
            'influenced': 'http://dbpedia.org/resource/Hans_Reichenbach',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Max_Born',
            'knownFor': "http://dbpedia.org/resource/Born_approximation|http://dbpedia.org/resource/Born_coordinates|http://dbpedia.org/resource/Born_equation|http://dbpedia.org/resource/Born_probability|http://dbpedia.org/resource/Born_rigidity|http://dbpedia.org/resource/Born–Haber_cycle|http://dbpedia.org/resource/Born–Huang_approximation|http://dbpedia.org/resource/Born–Infeld_theory|http://dbpedia.org/resource/Born–Landé_equation|http://dbpedia.org/resource/Born–Oppenheimer_approximation|http://dbpedia.org/resource/Born's_Rule|http://dbpedia.org/resource/Born–von_Karman_boundary_condition",
            'name': 'Max Born',
            'notableStudent': 'http://dbpedia.org/resource/Emil_Wolf',
            'residence': 'http://dbpedia.org/resource/Göttingen|http://dbpedia.org/resource/West_Germany',
            'resource': 'http://dbpedia.org/resource/Max_Born',
            'signature': 'Max Born signature.svg',
            'source': 'http://dbpedia.org/data/Max_Born.json',
            'spouse': 'Hedwig  Ehrenberg',
            'surname': 'Born',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Max_Born.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Max_Born?oldid=744208226',
            'wikiPageID': 61866,
            'wikiPageRevisionID': 744208226,
            'workplaces': 'http://dbpedia.org/resource/Johann_Wolfgang_Goethe_University_of_Frankfurt_am_Main|http://dbpedia.org/resource/University_of_Edinburgh|http://dbpedia.org/resource/University_of_Göttingen'}


@pytest.fixture
def expected_imputed_max_born_data():
    return {'abstract': 'Max Born (German: [bɔɐ̯n]; 11 December 1882 – 5 January 1970) '
            'was a German physicist and mathematician who was instrumental in '
            'the development of quantum mechanics. He also made contributions '
            'to solid-state physics and optics and supervised the work of a '
            'number of notable physicists in the 1920s and 1930s. Born won '
            'the 1954 Nobel Prize in Physics for his "fundamental research in '
            'Quantum Mechanics, especially in the statistical interpretation '
            'of the wave function". Born in 1882 in Breslau, then in Germany, '
            'now in Poland and known as Wrocław, Born entered the University '
            'of Göttingen in 1904, where he found the three renowned '
            'mathematicians, Felix Klein, David Hilbert and Hermann '
            'Minkowski. He wrote his Ph.D. thesis on the subject of '
            '"Stability of Elastica in a Plane and Space", winning the '
            "University's Philosophy Faculty Prize. In 1905, he began "
            'researching special relativity with Minkowski, and subsequently '
            'wrote his habilitation thesis on the Thomson model of the atom. '
            'A chance meeting with Fritz Haber in Berlin in 1918 led to '
            'discussion of the manner in which an ionic compound is formed '
            'when a metal reacts with a halogen, which is today known as the '
            'Born–Haber cycle. In the First World War, after originally being '
            'placed as a radio operator, he was moved to research duties '
            'regarding sound ranging due to his specialist knowledge. In '
            '1921, Born returned to Göttingen, arranging another chair for '
            'his long-time friend and colleague James Franck. Under Born, '
            "Göttingen became one of the world's foremost centres for "
            'physics. In 1925, Born and Werner Heisenberg formulated the '
            'matrix mechanics representation of quantum mechanics. The '
            'following year, he formulated the now-standard interpretation of '
            'the probability density function for ψ*ψ in the Schrödinger '
            'equation, for which he was awarded the Nobel Prize in 1954. His '
            'influence extended far beyond his own research. Max Delbrück, '
            'Siegfried Flügge, Friedrich Hund, Pascual Jordan, Maria '
            'Goeppert-Mayer, Lothar Wolfgang Nordheim, Robert Oppenheimer, '
            'and Victor Weisskopf all received their Ph.D. degrees under Born '
            'at Göttingen, and his assistants included Enrico Fermi, Werner '
            'Heisenberg, Gerhard Herzberg, Friedrich Hund, Pascual Jordan, '
            'Wolfgang Pauli, Léon Rosenfeld, Edward Teller, and Eugene '
            'Wigner. In January 1933, the Nazi Party came to power in '
            'Germany, and Born, who was Jewish, was suspended. He emigrated '
            "to Britain, where he took a job at St John's College, Cambridge, "
            'and wrote a popular science book, The Restless Universe, as well '
            'as Atomic Physics, which soon became a standard textbook. In '
            'October 1936, he became the Tait Professor of Natural Philosophy '
            'at the University of Edinburgh, where, working with German-born '
            'assistants E. Walter Kellermann and Klaus Fuchs, he continued '
            'his research into physics. Max Born became a naturalised British '
            'subject on 31 August 1939, one day before World War II broke out '
            'in Europe. He remained at Edinburgh until 1952. He retired to '
            'Bad Pyrmont, in West Germany, and died in a hospital in '
            'Göttingen on 5 January 1970.',
            'academicAdvisor': 'J. J. Thomson|Joseph Larmor|Karl Schwarzschild|Woldemar '
            'Voigt',
            'almaMater': 'University of Göttingen',
            'award': 'Fellow of the Royal Society|Hughes Medal|Max Planck Medal|Nobel '
            'Prize in Physics',
            'birthDate': '1882-12-11',
            'birthPlace': 'German Empire|Wrocław',
            'categories': '1882 births|1970 deaths|Academics of the University of '
            'Cambridge|Academics of the University of Edinburgh|Alumni of '
            'Gonville and Caius College, Cambridge|British Jews|British '
            'Lutherans|British people of German-Jewish descent|British '
            'physicists|Deists|Fellows of the Royal Society|Fellows of the '
            'Royal Society of Edinburgh|Foreign Members of the USSR Academy '
            'of Sciences|German emigrants to Scotland|German Jews|German '
            'Lutherans|German Nobel laureates|German physicists|Goethe '
            'University Frankfurt faculty|Grand Crosses with Star and Sash '
            'of the Order of Merit of the Federal Republic of '
            'Germany|Heidelberg University alumni|Honorary Members of the '
            'USSR Academy of Sciences|Humboldt University of Berlin '
            'faculty|Jewish physicists|Jews who immigrated to the United '
            'Kingdom to escape Nazism|Members of the Prussian Academy of '
            'Sciences|Nobel laureates in Physics|Optical physicists|People '
            'associated with the University of Zurich|People from the '
            'Province of Silesia|People from Wrocław|People who emigrated '
            'to escape Nazism|Quantum physicists|Scientists from '
            'Frankfurt|Silesian Jews|Theoretical physicists|University of '
            'Breslau alumni|University of Göttingen alumni|University of '
            'Göttingen faculty|Winners of the Max Planck Medal',
            'child': 'eldest  was mother of Olivia Newton-John',
            'citizenship': 'British|German',
            'comment': 'Max Born (German: [bɔɐ̯n]; 11 December 1882 – 5 January 1970) was '
            'a German physicist and mathematician who was instrumental in the '
            'development of quantum mechanics. He also made contributions to '
            'solid-state physics and optics and supervised the work of a '
            'number of notable physicists in the 1920s and 1930s. Born won the '
            '1954 Nobel Prize in Physics for his "fundamental research in '
            'Quantum Mechanics, especially in the statistical interpretation '
            'of the wave function".',
            'deathDate': '1970-01-05',
            'deathPlace': 'Göttingen|West Germany',
            'description': 'physicist',
            'doctoralAdvisor': 'Carl David Tolmé Runge',
            'doctoralStudent': 'Bertha Swirles|Cheng Kaijia|Edgar Krahn|Friedrich '
            'Hund|Herbert S. Green|J. Robert Oppenheimer|Lothar '
            'Wolfgang Nordheim|Maria Goeppert-Mayer|Maurice Pryce|Max '
            'Delbrück|Pascual Jordan|Peng Huanwu|Siegfried '
            'Flügge|Victor Weisskopf|Walter M. Elsasser',
            'field': 'Physics',
            'fullName': 'Max Born',
            'gender': 'male',
            'givenName': 'Max',
            'influenced': 'Hans Reichenbach',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Max_Born',
            'knownFor': 'Born approximation|Born coordinates|Born equation|Born '
            'rigidity|Born rule|Born–Haber cycle|Born–Huang '
            'approximation|Born–Infeld model|Born–Landé '
            'equation|Born–Oppenheimer approximation|Born–von Karman boundary '
            'condition|Probability amplitude',
            'name': 'Max Born',
            'notableStudent': 'Emil Wolf',
            'residence': 'Göttingen|West Germany',
            'resource': 'http://dbpedia.org/resource/Max_Born',
            'signature': 'Max Born signature.svg',
            'source': 'http://dbpedia.org/data/Max_Born.json',
            'spouse': 'Hedwig  Ehrenberg',
            'surname': 'Born',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Max_Born.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Max_Born?oldid=744208226',
            'wikiPageID': 61866,
            'wikiPageRevisionID': 744208226,
            'workplaces': 'Goethe University Frankfurt|University of Edinburgh|University '
            'of Göttingen'}


@pytest.fixture
def expected_niels_bohr_data():
    return {'abstract': 'Niels Henrik David Bohr (Danish: [nels ˈb̥oɐ̯ˀ]; 7 October 1885 '
            '– 18 November 1962) was a Danish physicist who made foundational '
            'contributions to understanding atomic structure and quantum '
            'theory, for which he received the Nobel Prize in Physics in '
            '1922. Bohr was also a philosopher and a promoter of scientific '
            'research. Bohr developed the Bohr model of the atom, in which he '
            'proposed that energy levels of electrons are discrete and that '
            'the electrons revolve in stable orbits around the atomic nucleus '
            'but can jump from one energy level (or orbit) to another. '
            'Although the Bohr model has been supplanted by other models, its '
            'underlying principles remain valid. He conceived the principle '
            'of complementarity: that items could be separately analysed in '
            'terms of contradictory properties, like behaving as a wave or a '
            'stream of particles. The notion of complementarity dominated '
            "Bohr's thinking in both science and philosophy. Bohr founded the "
            'Institute of Theoretical Physics at the University of '
            'Copenhagen, now known as the Niels Bohr Institute, which opened '
            'in 1920. Bohr mentored and collaborated with physicists '
            'including Hans Kramers, Oskar Klein, George de Hevesy and Werner '
            'Heisenberg. He predicted the existence of a new zirconium-like '
            'element, which was named hafnium, after the Latin name for '
            'Copenhagen, where it was discovered. Later, the element bohrium '
            'was named after him. During the 1930s, Bohr helped refugees from '
            'Nazism. After Denmark was occupied by the Germans, he had a '
            'famous meeting with Heisenberg, who had become the head of the '
            'German nuclear weapon project. In September 1943, word reached '
            'Bohr that he was about to be arrested by the Germans, and he '
            'fled to Sweden. From there, he was flown to Britain, where he '
            'joined the British Tube Alloys nuclear weapons project, and was '
            'part of the British mission to the Manhattan Project. After the '
            'war, Bohr called for international cooperation on nuclear '
            'energy. He was involved with the establishment of CERN and the '
            'Research Establishment Risø of the Danish Atomic Energy '
            'Commission, and became the first chairman of the Nordic '
            'Institute for Theoretical Physics in 1957.',
            'academicAdvisor': 'http://dbpedia.org/resource/Ernest_Rutherford|http://dbpedia.org/resource/J._J._Thomson',
            'almaMater': 'http://dbpedia.org/resource/Trinity_College,_Cambridge|http://dbpedia.org/resource/University_of_Copenhagen',
            'award': 'http://dbpedia.org/resource/Atoms_for_Peace_Award|http://dbpedia.org/resource/Copley_Medal|http://dbpedia.org/resource/Franklin_Medal|http://dbpedia.org/resource/Hughes_Medal|http://dbpedia.org/resource/Matteucci_Medal|http://dbpedia.org/resource/Max_Planck_Medal|http://dbpedia.org/resource/Nobel_Prize_in_Physics|http://dbpedia.org/resource/Order_of_the_Elephant|http://dbpedia.org/resource/Royal_Society|http://dbpedia.org/resource/Sonning_Prize',
            'birthDate': '1885-10-07',
            'birthName': 'Niels Henrik David Bohr',
            'birthPlace': 'http://dbpedia.org/resource/Copenhagen',
            'categories': 'http://dbpedia.org/resource/Category:1885_births|http://dbpedia.org/resource/Category:1962_deaths|http://dbpedia.org/resource/Category:20th-century_physicists|http://dbpedia.org/resource/Category:Academics_of_the_Victoria_University_of_Manchester|http://dbpedia.org/resource/Category:Akademisk_Boldklub_players|http://dbpedia.org/resource/Category:Alumni_of_Trinity_College,_Cambridge|http://dbpedia.org/resource/Category:Atoms_for_Peace_Award_recipients|http://dbpedia.org/resource/Category:Bohr_family|http://dbpedia.org/resource/Category:Corresponding_Members_of_the_Russian_Academy_of_Sciences_(1917–25)|http://dbpedia.org/resource/Category:Corresponding_Members_of_the_USSR_Academy_of_Sciences|http://dbpedia.org/resource/Category:Danish_atheists|http://dbpedia.org/resource/Category:Danish_expatriates_in_England|http://dbpedia.org/resource/Category:Danish_expatriates_in_the_United_States|http://dbpedia.org/resource/Category:Danish_footballers|http://dbpedia.org/resource/Category:Danish_Lutherans|http://dbpedia.org/resource/Category:Danish_Nobel_laureates|http://dbpedia.org/resource/Category:Danish_nuclear_physicists|http://dbpedia.org/resource/Category:Danish_people_of_Jewish_descent|http://dbpedia.org/resource/Category:Danish_people_of_World_War_II|http://dbpedia.org/resource/Category:Danish_philosophers|http://dbpedia.org/resource/Category:Danish_physicists|http://dbpedia.org/resource/Category:Faraday_Lecturers|http://dbpedia.org/resource/Category:Fellows_of_the_German_Academy_of_Sciences_Leopoldina|http://dbpedia.org/resource/Category:Foreign_Fellows_of_the_Indian_National_Science_Academy|http://dbpedia.org/resource/Category:Foreign_Members_of_the_Royal_Society|http://dbpedia.org/resource/Category:Grand_Crosses_of_the_Order_of_the_Dannebrog|http://dbpedia.org/resource/Category:Honorary_Members_of_the_USSR_Academy_of_Sciences|http://dbpedia.org/resource/Category:Institute_for_Advanced_Study_visiting_scholars|http://dbpedia.org/resource/Category:Manhattan_Project_people|http://dbpedia.org/resource/Category:Members_of_the_Pontifical_Academy_of_Sciences|http://dbpedia.org/resource/Category:Members_of_the_Prussian_Academy_of_Sciences|http://dbpedia.org/resource/Category:Members_of_the_Royal_Netherlands_Academy_of_Arts_and_Sciences|http://dbpedia.org/resource/Category:Niels_Bohr|http://dbpedia.org/resource/Category:Niels_Bohr_International_Gold_Medal_recipients|http://dbpedia.org/resource/Category:Nobel_laureates_in_Physics|http://dbpedia.org/resource/Category:People_associated_with_CERN|http://dbpedia.org/resource/Category:Philosophers_of_science|http://dbpedia.org/resource/Category:Quantum_physicists|http://dbpedia.org/resource/Category:Recipients_of_the_Copley_Medal|http://dbpedia.org/resource/Category:Recipients_of_the_Pour_le_Mérite_(civil_class)|http://dbpedia.org/resource/Category:Scientists_from_Copenhagen|http://dbpedia.org/resource/Category:Theoretical_physicists|http://dbpedia.org/resource/Category:University_of_Copenhagen_alumni|http://dbpedia.org/resource/Category:Winners_of_the_Max_Planck_Medal',
            'child': 'Aage Bohr|Ernest Bohr|four others',
            'comment': 'Niels Henrik David Bohr (Danish: [nels ˈb̥oɐ̯ˀ]; 7 October 1885 – '
            '18 November 1962) was a Danish physicist who made foundational '
            'contributions to understanding atomic structure and quantum '
            'theory, for which he received the Nobel Prize in Physics in 1922. '
            'Bohr was also a philosopher and a promoter of scientific '
            'research.',
            'deathDate': '1962-11-18',
            'deathPlace': 'http://dbpedia.org/resource/Copenhagen',
            'description': 'Danish physicist and receiver of a 1922 Nobel Prize',
            'doctoralAdvisor': 'http://dbpedia.org/resource/Christian_Christiansen',
            'doctoralStudent': 'http://dbpedia.org/resource/Hendrik_Anthony_Kramers',
            'field': 'http://dbpedia.org/resource/Physics',
            'fullName': 'Niels Bohr',
            'gender': 'male',
            'givenName': 'Niels',
            'influenced': 'http://dbpedia.org/resource/Bengt_Strömgren|http://dbpedia.org/resource/Jack_Abbott_(author)|http://dbpedia.org/resource/Lise_Meitner|http://dbpedia.org/resource/Max_Delbrück|http://dbpedia.org/resource/Menas_Kafatos|http://dbpedia.org/resource/Paul_Dirac|http://dbpedia.org/resource/Werner_Heisenberg|http://dbpedia.org/resource/Wolfgang_Pauli',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Niels_Bohr',
            'knownFor': 'http://dbpedia.org/resource/BKS_theory|http://dbpedia.org/resource/Bohr_magneton|http://dbpedia.org/resource/Bohr_model|http://dbpedia.org/resource/Bohr_orbital|http://dbpedia.org/resource/Bohr_radius|http://dbpedia.org/resource/Bohr–Einstein_debates|http://dbpedia.org/resource/Bohr–Sommerfeld_quantization|http://dbpedia.org/resource/Bohr–van_Leeuwen_theorem|http://dbpedia.org/resource/Complementarity_(physics)|http://dbpedia.org/resource/Copenhagen_interpretation|http://dbpedia.org/resource/Hafnium|http://dbpedia.org/resource/Sommerfeld–Bohr_theory',
            'name': 'Niels Bohr',
            'nationality': 'Danish',
            'notableStudent': 'http://dbpedia.org/resource/Lev_Landau',
            'resource': 'http://dbpedia.org/resource/Niels_Bohr',
            'signature': 'Niels Bohr Signature.svg',
            'source': 'http://dbpedia.org/data/Niels_Bohr.json',
            'spouse': 'Margrethe Nørlund',
            'surname': 'Bohr',
            'thesisTitle': 'Studier over Metallernes Elektrontheori',
            'thesisUrl': 'http://www.sciencedirect.com/science/article/pii/S187605030870015X',
            'thesisYear': 'May 1911',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Niels_Bohr.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Niels_Bohr?oldid=737858422',
            'wikiPageID': 21210,
            'wikiPageRevisionID': 737858422,
            'workplaces': 'http://dbpedia.org/resource/University_of_Cambridge|http://dbpedia.org/resource/University_of_Copenhagen|http://dbpedia.org/resource/Victoria_University_of_Manchester'
            }


@pytest.fixture
def expected_imputed_niels_bohr_data():
    return {'abstract': 'Niels Henrik David Bohr (Danish: [nels ˈb̥oɐ̯ˀ]; 7 October 1885 '
            '– 18 November 1962) was a Danish physicist who made foundational '
            'contributions to understanding atomic structure and quantum '
            'theory, for which he received the Nobel Prize in Physics in '
            '1922. Bohr was also a philosopher and a promoter of scientific '
            'research. Bohr developed the Bohr model of the atom, in which he '
            'proposed that energy levels of electrons are discrete and that '
            'the electrons revolve in stable orbits around the atomic nucleus '
            'but can jump from one energy level (or orbit) to another. '
            'Although the Bohr model has been supplanted by other models, its '
            'underlying principles remain valid. He conceived the principle '
            'of complementarity: that items could be separately analysed in '
            'terms of contradictory properties, like behaving as a wave or a '
            'stream of particles. The notion of complementarity dominated '
            "Bohr's thinking in both science and philosophy. Bohr founded the "
            'Institute of Theoretical Physics at the University of '
            'Copenhagen, now known as the Niels Bohr Institute, which opened '
            'in 1920. Bohr mentored and collaborated with physicists '
            'including Hans Kramers, Oskar Klein, George de Hevesy and Werner '
            'Heisenberg. He predicted the existence of a new zirconium-like '
            'element, which was named hafnium, after the Latin name for '
            'Copenhagen, where it was discovered. Later, the element bohrium '
            'was named after him. During the 1930s, Bohr helped refugees from '
            'Nazism. After Denmark was occupied by the Germans, he had a '
            'famous meeting with Heisenberg, who had become the head of the '
            'German nuclear weapon project. In September 1943, word reached '
            'Bohr that he was about to be arrested by the Germans, and he '
            'fled to Sweden. From there, he was flown to Britain, where he '
            'joined the British Tube Alloys nuclear weapons project, and was '
            'part of the British mission to the Manhattan Project. After the '
            'war, Bohr called for international cooperation on nuclear '
            'energy. He was involved with the establishment of CERN and the '
            'Research Establishment Risø of the Danish Atomic Energy '
            'Commission, and became the first chairman of the Nordic '
            'Institute for Theoretical Physics in 1957.',
            'academicAdvisor': 'Ernest Rutherford|J. J. Thomson',
            'almaMater': 'Trinity College, Cambridge|University of Copenhagen',
            'award': 'Atoms for Peace Award|Copley Medal|Franklin Medal|Hughes '
            'Medal|Matteucci Medal|Max Planck Medal|Nobel Prize in Physics|Order '
            'of the Elephant|Royal Society|Sonning Prize',
            'birthDate': '1885-10-07',
            'birthName': 'Niels Henrik David Bohr',
            'birthPlace': 'Copenhagen',
            'categories': '1885 births|1962 deaths|20th-century physicists|Academics of '
            'the Victoria University of Manchester|Akademisk Boldklub '
            'players|Alumni of Trinity College, Cambridge|Atoms for Peace '
            'Award recipients|Bohr family|Corresponding Members of the '
            'Russian Academy of Sciences (1917–25)|Corresponding Members of '
            'the USSR Academy of Sciences|Danish atheists|Danish '
            'expatriates in England|Danish expatriates in the United '
            'States|Danish footballers|Danish Lutherans|Danish Nobel '
            'laureates|Danish nuclear physicists|Danish people of Jewish '
            'descent|Danish people of World War II|Danish '
            'philosophers|Danish physicists|Faraday Lecturers|Fellows of '
            'the German Academy of Sciences Leopoldina|Foreign Fellows of '
            'the Indian National Science Academy|Foreign Members of the '
            'Royal Society|Grand Crosses of the Order of the '
            'Dannebrog|Honorary Members of the USSR Academy of '
            'Sciences|Institute for Advanced Study visiting '
            'scholars|Manhattan Project people|Members of the Pontifical '
            'Academy of Sciences|Members of the Prussian Academy of '
            'Sciences|Members of the Royal Netherlands Academy of Arts and '
            'Sciences|Niels Bohr|Niels Bohr International Gold Medal '
            'recipients|Nobel laureates in Physics|People associated with '
            'CERN|Philosophers of science|Quantum physicists|Recipients of '
            'the Copley Medal|Recipients of the Pour le Mérite (civil '
            'class)|Scientists from Copenhagen|Theoretical '
            'physicists|University of Copenhagen alumni|Winners of the Max '
            'Planck Medal',
            'child': 'Aage Bohr|Ernest Bohr|four others',
            'comment': 'Niels Henrik David Bohr (Danish: [nels ˈb̥oɐ̯ˀ]; 7 October 1885 – '
            '18 November 1962) was a Danish physicist who made foundational '
            'contributions to understanding atomic structure and quantum '
            'theory, for which he received the Nobel Prize in Physics in 1922. '
            'Bohr was also a philosopher and a promoter of scientific '
            'research.',
            'deathDate': '1962-11-18',
            'deathPlace': 'Copenhagen',
            'description': 'Danish physicist and receiver of a 1922 Nobel Prize',
            'doctoralAdvisor': 'Christian Christiansen',
            'doctoralStudent': 'Hans Kramers',
            'field': 'Physics',
            'fullName': 'Niels Bohr',
            'gender': 'male',
            'givenName': 'Niels',
            'influenced': 'Bengt Strömgren|Jack Abbott (author)|Lise Meitner|Max '
            'Delbrück|Menas Kafatos|Paul Dirac|Werner Heisenberg|Wolfgang '
            'Pauli',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Niels_Bohr',
            'knownFor': 'Atomic orbital|BKS theory|Bohr magneton|Bohr model|Bohr '
            'radius|Bohr–Einstein debates|Bohr–van Leeuwen '
            'theorem|Complementarity (physics)|Copenhagen '
            'interpretation|Hafnium|Old quantum theory',
            'name': 'Niels Bohr',
            'nationality': 'Danish',
            'notableStudent': 'Lev Landau',
            'resource': 'http://dbpedia.org/resource/Niels_Bohr',
            'signature': 'Niels Bohr Signature.svg',
            'source': 'http://dbpedia.org/data/Niels_Bohr.json',
            'spouse': 'Margrethe Nørlund',
            'surname': 'Bohr',
            'thesisTitle': 'Studier over Metallernes Elektrontheori',
            'thesisUrl': 'http://www.sciencedirect.com/science/article/pii/S187605030870015X',
            'thesisYear': 'May 1911',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Niels_Bohr.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Niels_Bohr?oldid=737858422',
            'wikiPageID': 21210,
            'wikiPageRevisionID': 737858422,
            'workplaces': 'University of Cambridge|University of Copenhagen|Victoria '
            'University of Manchester'}


@pytest.fixture
def expected_david_carroll_data():
    return {'abstract': 'David Carroll (born January 13, 1963) is a U.S. physicist, '
            'materials scientist and nanotechnologist, Fellow of the American '
            'Physical Society, and director of the Center for Nanotechnology '
            'and Molecular Materials at Wake Forest University. He has '
            'contributed to the field of nanoscience and nanotechnology '
            'through his work in nanoengineered cancer therapeutics, '
            'nanocomposite-based display and lighting technologies, high '
            'efficiency nanocomposite photovoltaics and thermo/piezo-electric '
            'generators.',
            'almaMater': 'North Carolina State University \n'
            'Wesleyan University \n'
            'University of Pennsylvania \n'
            'Max-Planck-Insitut für Metallforschung',
            'birthDate': '1963-01-13',
            'categories': 'http://dbpedia.org/resource/Category:1963_births|http://dbpedia.org/resource/Category:American_physicists|http://dbpedia.org/resource/Category:Clemson_University_faculty|http://dbpedia.org/resource/Category:Living_people|http://dbpedia.org/resource/Category:Nanotechnologists|http://dbpedia.org/resource/Category:North_Carolina_State_University_alumni|http://dbpedia.org/resource/Category:University_of_Pennsylvania|http://dbpedia.org/resource/Category:Wake_Forest_University_faculty|http://dbpedia.org/resource/Category:Wesleyan_University_alumni',
            'child': 'Lauren Carroll',
            'comment': 'David Carroll (born January 13, 1963) is a U.S. physicist, '
            'materials scientist and nanotechnologist, Fellow of the American '
            'Physical Society, and director of the Center for Nanotechnology '
            'and Molecular Materials at Wake Forest University. He has '
            'contributed to the field of nanoscience and nanotechnology '
            'through his work in nanoengineered cancer therapeutics, '
            'nanocomposite-based display and lighting technologies, high '
            'efficiency nanocomposite photovoltaics and thermo/piezo-electric '
            'generators.',
            'description': 'American physicist',
            'field': 'http://dbpedia.org/resource/Physics',
            'fullName': 'David Carroll (physicist)',
            'gender': 'male',
            'givenName': 'David',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/David_Carroll_(physicist)',
            'name': 'David Carroll',
            'residence': 'Winston-Salem, North Carolina, United States',
            'resource': 'http://dbpedia.org/resource/David_Carroll_(physicist)',
            'source': 'http://dbpedia.org/data/David_Carroll_(physicist).json',
            'spouse': 'Melissa Carroll',
            'surname': 'Carroll',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Physicist_David_Carroll.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/David_Carroll_(physicist)?oldid=740253932',
            'wikiPageID': 16738797,
            'wikiPageRevisionID': 740253932,
            'workplaces': 'Clemson University \nWake Forest University'}


@pytest.fixture
def expected_imputed_david_carroll_data():
    return {'abstract': 'David Carroll (born January 13, 1963) is a U.S. physicist, '
            'materials scientist and nanotechnologist, Fellow of the American '
            'Physical Society, and director of the Center for Nanotechnology '
            'and Molecular Materials at Wake Forest University. He has '
            'contributed to the field of nanoscience and nanotechnology '
            'through his work in nanoengineered cancer therapeutics, '
            'nanocomposite-based display and lighting technologies, high '
            'efficiency nanocomposite photovoltaics and thermo/piezo-electric '
            'generators.',
            'almaMater': 'Max-Planck-Insitut für Metallforschung|North Carolina State '
            'University|University of Pennsylvania|Wesleyan University',
            'birthDate': '1963-01-13',
            'categories': '1963 births|American physicists|Clemson University '
            'faculty|Living people|Nanotechnologists|North Carolina State '
            'University alumni|University of Pennsylvania|Wake Forest '
            'University faculty|Wesleyan University alumni',
            'child': 'Lauren Carroll',
            'comment': 'David Carroll (born January 13, 1963) is a U.S. physicist, '
            'materials scientist and nanotechnologist, Fellow of the American '
            'Physical Society, and director of the Center for Nanotechnology '
            'and Molecular Materials at Wake Forest University. He has '
            'contributed to the field of nanoscience and nanotechnology '
            'through his work in nanoengineered cancer therapeutics, '
            'nanocomposite-based display and lighting technologies, high '
            'efficiency nanocomposite photovoltaics and thermo/piezo-electric '
            'generators.',
            'description': 'American physicist',
            'field': 'Physics',
            'fullName': 'David Carroll (physicist)',
            'gender': 'male',
            'givenName': 'David',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/David_Carroll_(physicist)',
            'name': 'David Carroll',
            'residence': 'Winston-Salem, North Carolina, United States',
            'resource': 'http://dbpedia.org/resource/David_Carroll_(physicist)',
            'source': 'http://dbpedia.org/data/David_Carroll_(physicist).json',
            'spouse': 'Melissa Carroll',
            'surname': 'Carroll',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Physicist_David_Carroll.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/David_Carroll_(physicist)?oldid=740253932',
            'wikiPageID': 16738797,
            'wikiPageRevisionID': 740253932,
            'workplaces': 'Clemson University|Wake Forest University'}


@pytest.fixture
def expected_harvard_university_data():
    return {'abstract': 'Harvard University is a private, Ivy League research university '
            'in Cambridge, Massachusetts, established in 1636, whose history, '
            "influence, and wealth have made it one of the world's most "
            'prestigious universities. Established originally by the '
            'Massachusetts legislature and soon thereafter named for John '
            "Harvard (its first benefactor), Harvard is the United States' "
            'oldest institution of higher learning, and the Harvard '
            'Corporation (formally, the President and Fellows of Harvard '
            'College) is its first chartered corporation. Although never '
            'formally affiliated with any denomination, the early College '
            'primarily trained Congregationalist and Unitarian clergy. Its '
            'curriculum and student body were gradually secularized during '
            'the 18th century, and by the 19th century Harvard had emerged as '
            'the central cultural establishment among Boston elites. '
            "Following the American Civil War, President Charles W. Eliot's "
            'long tenure (1869–1909) transformed the college and affiliated '
            'professional schools into a modern research university; Harvard '
            'was a founding member of the Association of American '
            'Universities in 1900. James Bryant Conant led the university '
            'through the Great Depression and World War II and began to '
            'reform the curriculum and liberalize admissions after the war. '
            'The undergraduate college became coeducational after its 1977 '
            'merger with Radcliffe College. The University is organized into '
            'eleven separate academic units—ten faculties and the Radcliffe '
            'Institute for Advanced Study—with campuses throughout the Boston '
            'metropolitan area: its 209-acre (85 ha) main campus is centered '
            'on Harvard Yard in Cambridge, approximately 3 miles (5 km) '
            'northwest of Boston; the business school and athletics '
            'facilities, including Harvard Stadium, are located across the '
            'Charles River in the Allston neighborhood of Boston and the '
            'medical, dental, and public health schools are in the Longwood '
            "Medical Area. Harvard's $37.6 billion financial endowment is the "
            'largest of any academic institution. Harvard is a large, highly '
            'residential research university. The nominal cost of attendance '
            "is high, but the University's large endowment allows it to offer "
            'generous financial aid packages. It operates several arts, '
            'cultural, and scientific museums, alongside the Harvard Library, '
            "which is the world's largest academic and private library "
            'system, comprising 79 individual libraries with over 18 million '
            "volumes.Harvard's alumni include eight U.S. presidents, several "
            'foreign heads of state, 62 living billionaires, 335 Rhodes '
            'Scholars, and 242 Marshall Scholars. To date, some 150 Nobel '
            'laureates, 18 Fields Medalists and 13 Turing Award winners have '
            'been affiliated as students, faculty, or staff.',
            'categories': 'http://dbpedia.org/resource/Category:Harvard_University|http://dbpedia.org/resource/Category:V-12_Navy_College_Training_Program',
            'city': 'http://dbpedia.org/resource/Cambridge,_Massachusetts',
            'comment': 'Harvard University is a private, Ivy League research university '
            'in Cambridge, Massachusetts, established in 1636, whose history, '
            "influence, and wealth have made it one of the world's most "
            'prestigious universities.',
            'country': 'U.S.',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/Harvard_University_logo.PNG',
            'fullName': 'Harvard University',
            'homepage': 'http://www.harvard.edu/',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Harvard_University',
            'lat': 42.37444305419922,
            'long': -71.116943359375,
            'name': 'Harvard University',
            'resource': 'http://dbpedia.org/resource/Harvard_University',
            'source': 'http://dbpedia.org/data/Harvard_University.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Harvard_University_logo.PNG?width=300',
            'type': 'http://dbpedia.org/resource/Research_university',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Harvard_University?oldid=744688765',
            'wikiPageID': 18426501,
            'wikiPageRevisionID': 744688765}


@pytest.fixture
def expected_imputed_harvard_university_data():
    return {'abstract': 'Harvard University is a private, Ivy League research university '
            'in Cambridge, Massachusetts, established in 1636, whose history, '
            "influence, and wealth have made it one of the world's most "
            'prestigious universities. Established originally by the '
            'Massachusetts legislature and soon thereafter named for John '
            "Harvard (its first benefactor), Harvard is the United States' "
            'oldest institution of higher learning, and the Harvard '
            'Corporation (formally, the President and Fellows of Harvard '
            'College) is its first chartered corporation. Although never '
            'formally affiliated with any denomination, the early College '
            'primarily trained Congregationalist and Unitarian clergy. Its '
            'curriculum and student body were gradually secularized during '
            'the 18th century, and by the 19th century Harvard had emerged as '
            'the central cultural establishment among Boston elites. '
            "Following the American Civil War, President Charles W. Eliot's "
            'long tenure (1869–1909) transformed the college and affiliated '
            'professional schools into a modern research university; Harvard '
            'was a founding member of the Association of American '
            'Universities in 1900. James Bryant Conant led the university '
            'through the Great Depression and World War II and began to '
            'reform the curriculum and liberalize admissions after the war. '
            'The undergraduate college became coeducational after its 1977 '
            'merger with Radcliffe College. The University is organized into '
            'eleven separate academic units—ten faculties and the Radcliffe '
            'Institute for Advanced Study—with campuses throughout the Boston '
            'metropolitan area: its 209-acre (85 ha) main campus is centered '
            'on Harvard Yard in Cambridge, approximately 3 miles (5 km) '
            'northwest of Boston; the business school and athletics '
            'facilities, including Harvard Stadium, are located across the '
            'Charles River in the Allston neighborhood of Boston and the '
            'medical, dental, and public health schools are in the Longwood '
            "Medical Area. Harvard's $37.6 billion financial endowment is the "
            'largest of any academic institution. Harvard is a large, highly '
            'residential research university. The nominal cost of attendance '
            "is high, but the University's large endowment allows it to offer "
            'generous financial aid packages. It operates several arts, '
            'cultural, and scientific museums, alongside the Harvard Library, '
            "which is the world's largest academic and private library "
            'system, comprising 79 individual libraries with over 18 million '
            "volumes.Harvard's alumni include eight U.S. presidents, several "
            'foreign heads of state, 62 living billionaires, 335 Rhodes '
            'Scholars, and 242 Marshall Scholars. To date, some 150 Nobel '
            'laureates, 18 Fields Medalists and 13 Turing Award winners have '
            'been affiliated as students, faculty, or staff.',
            'categories': 'Harvard University|V-12 Navy College Training Program',
            'city': 'Cambridge, Massachusetts',
            'comment': 'Harvard University is a private, Ivy League research university '
            'in Cambridge, Massachusetts, established in 1636, whose history, '
            "influence, and wealth have made it one of the world's most "
            'prestigious universities.',
            'country': 'United States',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/Harvard_University_logo.PNG',
            'fullName': 'Harvard University',
            'homepage': 'http://www.harvard.edu/',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Harvard_University',
            'lat': 42.37444305419922,
            'long': -71.116943359375,
            'name': 'Harvard University',
            'resource': 'http://dbpedia.org/resource/Harvard_University',
            'source': 'http://dbpedia.org/data/Harvard_University.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Harvard_University_logo.PNG?width=300',
            'type': 'University',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Harvard_University?oldid=744688765',
            'wikiPageID': 18426501,
            'wikiPageRevisionID': 744688765}


@pytest.fixture
def expected_institute_for_quantum_computing_data():
    return {'abstract': 'The Institute for Quantum Computing, or IQC, located in '
            'Waterloo, Ontario, is an affiliate scientific research institute '
            'of the University of Waterloo with a multidisciplinary approach '
            'to the field of quantum information processing. IQC was founded '
            'in 2002 primarily through a donation made by Mike Lazaridis and '
            'his wife Ophelia whose substantial donations have continued over '
            'the years. The institute is now located in the Mike & Ophelia '
            'Lazaridis Quantum-Nano Centre and the Research Advancement '
            'Centre at the University of Waterloo. It is led by co-founder '
            'and physicist, Raymond Laflamme with researchers based in 6 '
            'departments across 3 faculties at the University of Waterloo. In '
            'addition to theoretical and experimental research on quantum '
            'computing, IQC also hosts academic conferences and workshops, '
            'short courses for undergraduate and high school students, and '
            'scientific outreach events including open houses and tours for '
            'the public.',
            'categories': 'http://dbpedia.org/resource/Category:2002_establishments_in_Ontario|http://dbpedia.org/resource/Category:Computer_science_institutes_in_Canada|http://dbpedia.org/resource/Category:Quantum_information_science|http://dbpedia.org/resource/Category:Research_institutes_in_Canada|http://dbpedia.org/resource/Category:University_of_Waterloo',
            'city': 'http://dbpedia.org/resource/Waterloo,_Ontario',
            'comment': 'The Institute for Quantum Computing, or IQC, located in Waterloo, '
            'Ontario, is an affiliate scientific research institute of the '
            'University of Waterloo with a multidisciplinary approach to the '
            'field of quantum information processing. IQC was founded in 2002 '
            'primarily through a donation made by Mike Lazaridis and his wife '
            'Ophelia whose substantial donations have continued over the '
            'years. The institute is now located in the Mike & Ophelia '
            'Lazaridis Quantum-Nano Centre and the Research Advancement Centre '
            'at the University of Waterloo.',
            'country': 'http://dbpedia.org/resource/Canada',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/IQC_logo,_updated_2013.png',
            'fullName': 'Institute for Quantum Computing',
            'homepage': 'http://uwaterloo.ca/institute-for-quantum-computing/',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Institute_for_Quantum_Computing',
            'lat': 43.47886657714844,
            'long': -80.55485534667969,
            'name': 'Institute for Quantum Computing (IQC)',
            'resource': 'http://dbpedia.org/resource/Institute_for_Quantum_Computing',
            'source': 'http://dbpedia.org/data/Institute_for_Quantum_Computing.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/IQC_logo,_updated_2013.png?width=300',
            'type': 'http://dbpedia.org/resource/Research_institute',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Institute_for_Quantum_Computing?oldid=725104800',
            'wikiPageID': 7952833,
            'wikiPageRevisionID': 725104800}


@pytest.fixture
def expected_imputed_institute_for_quantum_computing_data():
    return {'abstract': 'The Institute for Quantum Computing, or IQC, located in '
            'Waterloo, Ontario, is an affiliate scientific research institute '
            'of the University of Waterloo with a multidisciplinary approach '
            'to the field of quantum information processing. IQC was founded '
            'in 2002 primarily through a donation made by Mike Lazaridis and '
            'his wife Ophelia whose substantial donations have continued over '
            'the years. The institute is now located in the Mike & Ophelia '
            'Lazaridis Quantum-Nano Centre and the Research Advancement '
            'Centre at the University of Waterloo. It is led by co-founder '
            'and physicist, Raymond Laflamme with researchers based in 6 '
            'departments across 3 faculties at the University of Waterloo. In '
            'addition to theoretical and experimental research on quantum '
            'computing, IQC also hosts academic conferences and workshops, '
            'short courses for undergraduate and high school students, and '
            'scientific outreach events including open houses and tours for '
            'the public.',
            'categories': '2002 establishments in Ontario|Computer science institutes in '
            'Canada|Quantum information science|Research institutes in '
            'Canada|University of Waterloo',
            'city': 'Waterloo, Ontario',
            'comment': 'The Institute for Quantum Computing, or IQC, located in Waterloo, '
            'Ontario, is an affiliate scientific research institute of the '
            'University of Waterloo with a multidisciplinary approach to the '
            'field of quantum information processing. IQC was founded in 2002 '
            'primarily through a donation made by Mike Lazaridis and his wife '
            'Ophelia whose substantial donations have continued over the '
            'years. The institute is now located in the Mike & Ophelia '
            'Lazaridis Quantum-Nano Centre and the Research Advancement Centre '
            'at the University of Waterloo.',
            'country': 'Canada',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/IQC_logo,_updated_2013.png',
            'fullName': 'Institute for Quantum Computing',
            'homepage': 'http://uwaterloo.ca/institute-for-quantum-computing/',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Institute_for_Quantum_Computing',
            'lat': 43.47886657714844,
            'long': -80.55485534667969,
            'name': 'Institute for Quantum Computing (IQC)',
            'resource': 'http://dbpedia.org/resource/Institute_for_Quantum_Computing',
            'source': 'http://dbpedia.org/data/Institute_for_Quantum_Computing.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/IQC_logo,_updated_2013.png?width=300',
            'type': 'Research institute',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Institute_for_Quantum_Computing?oldid=725104800',
            'wikiPageID': 7952833,
            'wikiPageRevisionID': 725104800}


@pytest.fixture
def expected_multiple_country_url_data():
    return {'abstract': 'Belfast (/ˈbɛl.fɑːst/ or /ˈbɛl.fæst/; from Irish: Béal Feirste, '
            'meaning "mouth of the sandbanks") is the capital and largest '
            'city of Northern Ireland, and the centre of the tenth largest '
            'Primary Urban Area in the United Kingdom. On the River Lagan, it '
            'had a population of 286,000 at the 2011 census and 333,871 after '
            'the 2015 council reform. Belfast was granted city status in '
            '1888. Belfast was a centre of the Irish linen, tobacco '
            'processing, rope-making and shipbuilding industries: in the '
            'early 20th century, Harland and Wolff, which built the RMS '
            "Titanic, was the world's biggest and most productive shipyard. "
            'Belfast played a key role in the Industrial Revolution, and was '
            'a global industrial centre until the latter half of the 20th '
            'century. It has sustained a major aerospace and missiles '
            'industry since the mid 1930s. Industrialisation and the inward '
            "migration it brought made Belfast Ireland's biggest at the "
            'beginning of the 20th century. Today, Belfast remains a centre '
            'for industry, as well as the arts, higher education, business, '
            'and law, and is the economic engine of Northern Ireland. The '
            'city suffered greatly during the conflict called "the Troubles", '
            'but latterly has undergone a sustained period of calm, free from '
            'the intense political violence of former years, and substantial '
            'economic and commercial growth. Additionally, Belfast city '
            'centre has undergone considerable expansion and regeneration in '
            'recent years, notably around Victoria Square. Belfast is served '
            'by two airports: George Best Belfast City Airport in the city, '
            'and Belfast International Airport 15 miles (24 km) west of the '
            'city. Belfast is a major port, with commercial and industrial '
            'docks dominating the Belfast Lough shoreline, including the '
            'Harland and Wolff shipyard, and is listed by the Globalization '
            'and World Cities Research Network (GaWC) as a global city.',
            'categories': 'http://dbpedia.org/resource/Category:Belfast|http://dbpedia.org/resource/Category:British_capitals|http://dbpedia.org/resource/Category:Capitals_in_Europe|http://dbpedia.org/resource/Category:Districts_of_Northern_Ireland,_1972–2015|http://dbpedia.org/resource/Category:Districts_of_Northern_Ireland,_2015-present|http://dbpedia.org/resource/Category:Populated_coastal_places_in_the_United_Kingdom|http://dbpedia.org/resource/Category:Port_cities_and_towns_in_Northern_Ireland|http://dbpedia.org/resource/Category:Port_cities_and_towns_of_the_Irish_Sea|http://dbpedia.org/resource/Category:University_towns_in_Ireland|http://dbpedia.org/resource/Category:University_towns_in_the_United_Kingdom',
            'comment': 'Belfast (/ˈbɛl.fɑːst/ or /ˈbɛl.fæst/; from Irish: Béal Feirste, '
            'meaning "mouth of the sandbanks") is the capital and largest city '
            'of Northern Ireland, and the centre of the tenth largest Primary '
            'Urban Area in the United Kingdom. On the River Lagan, it had a '
            'population of 286,000 at the 2011 census and 333,871 after the '
            '2015 council reform. Belfast was granted city status in 1888.',
            'country': 'http://dbpedia.org/resource/Northern_Ireland|http://dbpedia.org/resource/United_Kingdom',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/NewBelfastCollage.jpg',
            'fullName': 'Belfast',
            'homepage': 'http://www.belfastcity.gov.uk/',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Belfast',
            'lat': 54.59700012207031,
            'long': -5.929166793823242,
            'name': 'Béal Feirste|Belfast|Bilfawst/Bilfaust',
            'resource': 'http://dbpedia.org/resource/Belfast',
            'source': 'http://dbpedia.org/data/Belfast.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/NewBelfastCollage.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Belfast?oldid=745101579',
            'wikiPageID': 5046,
            'wikiPageRevisionID': 745101579}


@pytest.fixture
def expected_imputed_multiple_country_url_data():
    return {'abstract': 'Belfast (/ˈbɛl.fɑːst/ or /ˈbɛl.fæst/; from Irish: Béal Feirste, '
            'meaning "mouth of the sandbanks") is the capital and largest '
            'city of Northern Ireland, and the centre of the tenth largest '
            'Primary Urban Area in the United Kingdom. On the River Lagan, it '
            'had a population of 286,000 at the 2011 census and 333,871 after '
            'the 2015 council reform. Belfast was granted city status in '
            '1888. Belfast was a centre of the Irish linen, tobacco '
            'processing, rope-making and shipbuilding industries: in the '
            'early 20th century, Harland and Wolff, which built the RMS '
            "Titanic, was the world's biggest and most productive shipyard. "
            'Belfast played a key role in the Industrial Revolution, and was '
            'a global industrial centre until the latter half of the 20th '
            'century. It has sustained a major aerospace and missiles '
            'industry since the mid 1930s. Industrialisation and the inward '
            "migration it brought made Belfast Ireland's biggest at the "
            'beginning of the 20th century. Today, Belfast remains a centre '
            'for industry, as well as the arts, higher education, business, '
            'and law, and is the economic engine of Northern Ireland. The '
            'city suffered greatly during the conflict called "the Troubles", '
            'but latterly has undergone a sustained period of calm, free from '
            'the intense political violence of former years, and substantial '
            'economic and commercial growth. Additionally, Belfast city '
            'centre has undergone considerable expansion and regeneration in '
            'recent years, notably around Victoria Square. Belfast is served '
            'by two airports: George Best Belfast City Airport in the city, '
            'and Belfast International Airport 15 miles (24 km) west of the '
            'city. Belfast is a major port, with commercial and industrial '
            'docks dominating the Belfast Lough shoreline, including the '
            'Harland and Wolff shipyard, and is listed by the Globalization '
            'and World Cities Research Network (GaWC) as a global city.',
            'categories': 'Belfast|British capitals|Capitals in Europe|Districts of '
            'Northern Ireland, 1972–2015|Districts of Northern Ireland, '
            '2015-present|Populated coastal places in the United '
            'Kingdom|Port cities and towns in Northern Ireland|Port cities '
            'and towns of the Irish Sea|University towns in '
            'Ireland|University towns in the United Kingdom',
            'comment': 'Belfast (/ˈbɛl.fɑːst/ or /ˈbɛl.fæst/; from Irish: Béal Feirste, '
            'meaning "mouth of the sandbanks") is the capital and largest city '
            'of Northern Ireland, and the centre of the tenth largest Primary '
            'Urban Area in the United Kingdom. On the River Lagan, it had a '
            'population of 286,000 at the 2011 census and 333,871 after the '
            '2015 council reform. Belfast was granted city status in 1888.',
            'country': 'Northern Ireland|United Kingdom',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/NewBelfastCollage.jpg',
            'fullName': 'Belfast',
            'homepage': 'http://www.belfastcity.gov.uk/',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Belfast',
            'lat': 54.59700012207031,
            'long': -5.929166793823242,
            'name': 'Béal Feirste|Belfast|Bilfawst/Bilfaust',
            'resource': 'http://dbpedia.org/resource/Belfast',
            'source': 'http://dbpedia.org/data/Belfast.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/NewBelfastCollage.jpg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Belfast?oldid=745101579',
            'wikiPageID': 5046,
            'wikiPageRevisionID': 745101579}


@pytest.fixture
def expected_multiple_country_no_url_data():
    return {'abstract': 'The Kingdom of Prussia (German: Königreich Preußen) was a German '
            'kingdom that constituted the state of Prussia between 1701 and '
            '1918 and included parts of present-day Germany, Poland, Russia, '
            'Lithuania, Denmark, Belgium and the Czech Republic. It was the '
            'driving force behind the unification of Germany in 1871 and was '
            'the leading state of the German Empire until its dissolution in '
            '1918. Although it took its name from the region called Prussia, '
            'it was based in the Margraviate of Brandenburg, where its '
            'capital was Berlin. The kings of Prussia were from the House of '
            'Hohenzollern. Prussia was a great power from the time it became '
            'a kingdom, through its predecessor, Brandenburg-Prussia, which '
            'became a military power under Frederick William, known as "The '
            'Great Elector". Prussia continued its rise to power under the '
            'guidance of Frederick II, more commonly known as Frederick the '
            'Great, the third son of Frederick William I. Frederick the Great '
            "was instrumental in starting the Seven Years' War, holding his "
            'own against Austria, Russia, France and Sweden and establishing '
            'Prussia’s role in the German states, as well as establishing the '
            'country as a European great power. After the might of Prussia '
            'was revealed it became a major power among the German states. '
            'Throughout the next hundred years Prussia went on to win many '
            'battles. It was because of its power that Prussia continuously '
            'tried to unify all the German states under its rule. After the '
            'Napoleonic Wars the issue of unifying Germany into one country '
            'caused revolution throughout the German states, with each '
            'wanting their own constitution. Initial attempts to unite '
            'Germany were unsuccessful, until the time of the North German '
            'Confederation which lasted from 1867–1871. It was seen as more '
            'of an alliance of military strength in the aftermath of the '
            'Austro-Prussian War but many of its laws were later used in the '
            'German Empire. The German Empire lasted from 1871–1918 with the '
            'successful unification of all the German states under Prussian '
            'hegemony. This was due to the defeat of Napoleon III in the '
            'Franco-Prussian War of 1870–71. The war united all the German '
            'states against a common enemy, and with the victory came an '
            'overwhelming wave of patriotism which changed the opinions of '
            'some of those who had been against unification. In 1871, Germany '
            'unified into a single country, minus Austria and Switzerland, '
            'with Prussia the dominant power. Prussia is considered the legal '
            'predecessor of the unified German Reich (1871–1945) and as such '
            "a direct ancestor of today's Federal Republic of Germany. The "
            'formal abolition of Prussia, carried out on 25 February 1947 by '
            'the fiat of the Allied Control Council referred to an alleged '
            'tradition of the kingdom as a bearer of militarism and reaction, '
            'and made way for the current setup of the German states. '
            'However, the Free State of Prussia (Freistaat Preußen), which '
            'followed the abolition of the Kingdom of Prussia in the '
            'aftermath of World War I, was a major democratic force in Weimar '
            'Germany until the nationalist coup of 1932 known as the '
            'Preußenschlag. The Kingdom left a significant cultural legacy, '
            'today notably promoted by the Prussian Cultural Heritage '
            'Foundation (Stiftung Preußischer Kulturbesitz (SPK)), which has '
            'become one of the largest cultural organisations in the world.',
            'categories': 'http://dbpedia.org/resource/Category:1701_establishments_in_Prussia|http://dbpedia.org/resource/Category:1918_disestablishments_in_Prussia|http://dbpedia.org/resource/Category:Former_kingdoms|http://dbpedia.org/resource/Category:Kingdom_of_Prussia|http://dbpedia.org/resource/Category:States_and_territories_disestablished_in_1918|http://dbpedia.org/resource/Category:States_and_territories_established_in_1701|http://dbpedia.org/resource/Category:States_of_the_German_Confederation|http://dbpedia.org/resource/Category:States_of_the_German_Empire|http://dbpedia.org/resource/Category:States_of_the_North_German_Confederation',
            'comment': 'The Kingdom of Prussia (German: Königreich Preußen) was a German '
            'kingdom that constituted the state of Prussia between 1701 and '
            '1918 and included parts of present-day Germany, Poland, Russia, '
            'Lithuania, Denmark, Belgium and the Czech Republic. It was the '
            'driving force behind the unification of Germany in 1871 and was '
            'the leading state of the German Empire until its dissolution in '
            '1918. Although it took its name from the region called Prussia, '
            'it was based in the Margraviate of Brandenburg, where its capital '
            'was Berlin.',
            'country': 'Belgium|Czech Republic were formally parts of '
            'Prussia.|Denmark|Germany|Lithuania|Poland|Russia',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Prussia_1892-1918.svg',
            'fullName': 'Kingdom of Prussia',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Kingdom_of_Prussia',
            'lat': 52.51666641235352,
            'long': 13.39999961853027,
            'name': 'Kingdom of Prussia|Prussia',
            'resource': 'http://dbpedia.org/resource/Kingdom_of_Prussia',
            'source': 'http://dbpedia.org/data/Kingdom_of_Prussia.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Prussia_1892-1918.svg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Kingdom_of_Prussia?oldid=744341596',
            'wikiPageID': 242701,
            'wikiPageRevisionID': 744341596}


@pytest.fixture
def expected_imputed_multiple_country_no_url_data():
    return {'abstract': 'The Kingdom of Prussia (German: Königreich Preußen) was a German '
            'kingdom that constituted the state of Prussia between 1701 and '
            '1918 and included parts of present-day Germany, Poland, Russia, '
            'Lithuania, Denmark, Belgium and the Czech Republic. It was the '
            'driving force behind the unification of Germany in 1871 and was '
            'the leading state of the German Empire until its dissolution in '
            '1918. Although it took its name from the region called Prussia, '
            'it was based in the Margraviate of Brandenburg, where its '
            'capital was Berlin. The kings of Prussia were from the House of '
            'Hohenzollern. Prussia was a great power from the time it became '
            'a kingdom, through its predecessor, Brandenburg-Prussia, which '
            'became a military power under Frederick William, known as "The '
            'Great Elector". Prussia continued its rise to power under the '
            'guidance of Frederick II, more commonly known as Frederick the '
            'Great, the third son of Frederick William I. Frederick the Great '
            "was instrumental in starting the Seven Years' War, holding his "
            'own against Austria, Russia, France and Sweden and establishing '
            'Prussia’s role in the German states, as well as establishing the '
            'country as a European great power. After the might of Prussia '
            'was revealed it became a major power among the German states. '
            'Throughout the next hundred years Prussia went on to win many '
            'battles. It was because of its power that Prussia continuously '
            'tried to unify all the German states under its rule. After the '
            'Napoleonic Wars the issue of unifying Germany into one country '
            'caused revolution throughout the German states, with each '
            'wanting their own constitution. Initial attempts to unite '
            'Germany were unsuccessful, until the time of the North German '
            'Confederation which lasted from 1867–1871. It was seen as more '
            'of an alliance of military strength in the aftermath of the '
            'Austro-Prussian War but many of its laws were later used in the '
            'German Empire. The German Empire lasted from 1871–1918 with the '
            'successful unification of all the German states under Prussian '
            'hegemony. This was due to the defeat of Napoleon III in the '
            'Franco-Prussian War of 1870–71. The war united all the German '
            'states against a common enemy, and with the victory came an '
            'overwhelming wave of patriotism which changed the opinions of '
            'some of those who had been against unification. In 1871, Germany '
            'unified into a single country, minus Austria and Switzerland, '
            'with Prussia the dominant power. Prussia is considered the legal '
            'predecessor of the unified German Reich (1871–1945) and as such '
            "a direct ancestor of today's Federal Republic of Germany. The "
            'formal abolition of Prussia, carried out on 25 February 1947 by '
            'the fiat of the Allied Control Council referred to an alleged '
            'tradition of the kingdom as a bearer of militarism and reaction, '
            'and made way for the current setup of the German states. '
            'However, the Free State of Prussia (Freistaat Preußen), which '
            'followed the abolition of the Kingdom of Prussia in the '
            'aftermath of World War I, was a major democratic force in Weimar '
            'Germany until the nationalist coup of 1932 known as the '
            'Preußenschlag. The Kingdom left a significant cultural legacy, '
            'today notably promoted by the Prussian Cultural Heritage '
            'Foundation (Stiftung Preußischer Kulturbesitz (SPK)), which has '
            'become one of the largest cultural organisations in the world.',
            'categories': '1701 establishments in Prussia|1918 disestablishments in '
            'Prussia|Former kingdoms|Kingdom of Prussia|States and '
            'territories disestablished in 1918|States and territories '
            'established in 1701|States of the German Confederation|States '
            'of the German Empire|States of the North German Confederation',
            'comment': 'The Kingdom of Prussia (German: Königreich Preußen) was a German '
            'kingdom that constituted the state of Prussia between 1701 and '
            '1918 and included parts of present-day Germany, Poland, Russia, '
            'Lithuania, Denmark, Belgium and the Czech Republic. It was the '
            'driving force behind the unification of Germany in 1871 and was '
            'the leading state of the German Empire until its dissolution in '
            '1918. Although it took its name from the region called Prussia, '
            'it was based in the Margraviate of Brandenburg, where its capital '
            'was Berlin.',
            'country': 'Belgium|Czech Republic were formally parts of '
            'Prussia.|Denmark|Germany|Lithuania|Poland|Russia',
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Prussia_1892-1918.svg',
            'fullName': 'Kingdom of Prussia',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Kingdom_of_Prussia',
            'lat': 52.51666641235352,
            'long': 13.39999961853027,
            'name': 'Kingdom of Prussia|Prussia',
            'resource': 'http://dbpedia.org/resource/Kingdom_of_Prussia',
            'source': 'http://dbpedia.org/data/Kingdom_of_Prussia.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Prussia_1892-1918.svg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Kingdom_of_Prussia?oldid=744341596',
            'wikiPageID': 242701,
            'wikiPageRevisionID': 744341596}


@pytest.fixture
def expected_two_lat_long_data():
    return {'abstract': 'Greece (Greek: Ελλάδα,  [eˈlaða]), officially the Hellenic '
            'Republic (Greek: Ελληνική Δημοκρατία Ellīnikī́ Dīmokratía '
            '[eliniˈci ðimokraˈti.a]), also known since ancient times as '
            'Hellas (Ancient Greek: Ἑλλάς Hellás [ˈhɛləs]), is a '
            'transcontinental country located in southeastern Europe. '
            "Greece's population is approximately 10.9 million as of 2015. "
            "Athens is the nation's capital and largest city, followed by "
            'Thessaloniki. Greece is strategically located at the crossroads '
            'of Europe, Asia, and Africa. Situated on the southern tip of the '
            'Balkan peninsula, it shares land borders with Albania to the '
            'northwest, the Republic of Macedonia and Bulgaria to the north, '
            'and Turkey to the northeast. Greece consists of nine geographic '
            'regions: Macedonia, Central Greece, the Peloponnese, Thessaly, '
            'Epirus, the Aegean Islands (including the Dodecanese and '
            'Cyclades), Thrace, Crete, and the Ionian Islands. The Aegean Sea '
            'lies to the east of the mainland, the Ionian Sea to the west, '
            'the Cretan Sea and the Mediterranean Sea to the south. Greece '
            'has the longest coastline on the Mediterranean Basin and the '
            '11th longest coastline in the world at 13,676 km (8,498 mi) in '
            'length, featuring a vast number of islands, of which 227 are '
            'inhabited. Eighty percent of Greece is mountainous, with Mount '
            'Olympus being the highest peak at 2,918 metres (9,573 ft). The '
            'history of Greece is one of the longest of any country, having '
            'been continuously inhabited since 270,000 BC. Considered the '
            'cradle of Western civilization, Greece is the birthplace of '
            'democracy, Western philosophy, the Olympic Games, Western '
            'literature, historiography, political science, major scientific '
            'and mathematical principles, and Western drama, including both '
            'tragedy and comedy. From the eighth century BC, the Greeks were '
            'organised into various independent city-states, known as polis, '
            'which spanned the entire Mediterranean region and the Black Sea. '
            'Philip of Macedon united most of the Greek mainland in the '
            'fourth century BC, with his son Alexander the Great rapidly '
            'conquering much of the ancient world, spreading Greek culture '
            'and science from the eastern Mediterranean to the Indus River. '
            'Greece was annexed by Rome in the second century BC, becoming an '
            'integral part of the Roman Empire and its successor, the '
            'Byzantine Empire, wherein the Greek language and culture were '
            'dominant. The establishment of the Greek Orthodox Church in the '
            'first century AD shaped modern Greek identity and transmitted '
            'Greek traditions to the wider Orthodox World. Falling under '
            'Ottoman dominion in the mid-15th century, the modern nation '
            'state of Greece emerged in 1830 following a war of independence. '
            "Greece's rich historical legacy is reflected by its 18 UNESCO "
            'World Heritage Sites, among the most in Europe and the world. '
            'Greece is a democratic and developed country with an advanced '
            'high-income economy, a high quality of life, and a very high '
            'standard of living. A founding member of the United Nations, '
            'Greece was the tenth member to join the European Communities '
            '(precursor to the European Union) and has been part of the '
            'Eurozone since 2001. It is also a member of numerous other '
            'international institutions, including the Council of Europe, the '
            'North Atlantic Treaty Organization (NATO), the Organisation for '
            'Economic Co-operation and Development (OECD), the World Trade '
            'Organization (WTO), the Organization for Security and '
            'Co-operation in Europe (OSCE), and the Organisation '
            "internationale de la Francophonie (OIF). Greece's unique "
            'cultural heritage, large tourism industry, prominent shipping '
            'sector and geostrategic importance classify it as a middle '
            'power. It is the largest economy in the Balkans, where it is an '
            'important regional investor.',
            'categories': 'http://dbpedia.org/resource/Category:1821_establishments_in_Europe|http://dbpedia.org/resource/Category:Balkan_countries|http://dbpedia.org/resource/Category:Countries_in_Europe|http://dbpedia.org/resource/Category:Greece|http://dbpedia.org/resource/Category:Liberal_democracies|http://dbpedia.org/resource/Category:Member_states_of_NATO|http://dbpedia.org/resource/Category:Member_states_of_the_Council_of_Europe|http://dbpedia.org/resource/Category:Member_states_of_the_European_Union|http://dbpedia.org/resource/Category:Member_states_of_the_Organisation_internationale_de_la_Francophonie|http://dbpedia.org/resource/Category:Member_states_of_the_Union_for_the_Mediterranean|http://dbpedia.org/resource/Category:Member_states_of_the_United_Nations|http://dbpedia.org/resource/Category:Republics|http://dbpedia.org/resource/Category:Southeastern_European_countries|http://dbpedia.org/resource/Category:Southern_European_countries|http://dbpedia.org/resource/Category:States_and_territories_established_in_1821',
            'comment': 'Greece (Greek: Ελλάδα,  [eˈlaða]), officially the Hellenic '
            'Republic (Greek: Ελληνική Δημοκρατία Ellīnikī́ Dīmokratía '
            '[eliniˈci ðimokraˈti.a]), also known since ancient times as '
            'Hellas (Ancient Greek: Ἑλλάς Hellás [ˈhɛləs]), is a '
            "transcontinental country located in southeastern Europe. Greece's "
            'population is approximately 10.9 million as of 2015. Athens is '
            "the nation's capital and largest city, followed by Thessaloniki.",
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Greece.svg',
            'fullName': 'Greece',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Greece',
            'lat': 37.96666717529297,
            'long': 23.71666717529297,
            'name': 'Greece',
            'resource': 'http://dbpedia.org/resource/Greece',
            'source': 'http://dbpedia.org/data/Greece.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Greece.svg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Greece?oldid=744805729',
            'wikiPageID': 12108,
            'wikiPageRevisionID': 744805729}


@pytest.fixture
def expected_imputed_two_lat_long_data():
    return {'abstract': 'Greece (Greek: Ελλάδα,  [eˈlaða]), officially the Hellenic '
            'Republic (Greek: Ελληνική Δημοκρατία Ellīnikī́ Dīmokratía '
            '[eliniˈci ðimokraˈti.a]), also known since ancient times as '
            'Hellas (Ancient Greek: Ἑλλάς Hellás [ˈhɛləs]), is a '
            'transcontinental country located in southeastern Europe. '
            "Greece's population is approximately 10.9 million as of 2015. "
            "Athens is the nation's capital and largest city, followed by "
            'Thessaloniki. Greece is strategically located at the crossroads '
            'of Europe, Asia, and Africa. Situated on the southern tip of the '
            'Balkan peninsula, it shares land borders with Albania to the '
            'northwest, the Republic of Macedonia and Bulgaria to the north, '
            'and Turkey to the northeast. Greece consists of nine geographic '
            'regions: Macedonia, Central Greece, the Peloponnese, Thessaly, '
            'Epirus, the Aegean Islands (including the Dodecanese and '
            'Cyclades), Thrace, Crete, and the Ionian Islands. The Aegean Sea '
            'lies to the east of the mainland, the Ionian Sea to the west, '
            'the Cretan Sea and the Mediterranean Sea to the south. Greece '
            'has the longest coastline on the Mediterranean Basin and the '
            '11th longest coastline in the world at 13,676 km (8,498 mi) in '
            'length, featuring a vast number of islands, of which 227 are '
            'inhabited. Eighty percent of Greece is mountainous, with Mount '
            'Olympus being the highest peak at 2,918 metres (9,573 ft). The '
            'history of Greece is one of the longest of any country, having '
            'been continuously inhabited since 270,000 BC. Considered the '
            'cradle of Western civilization, Greece is the birthplace of '
            'democracy, Western philosophy, the Olympic Games, Western '
            'literature, historiography, political science, major scientific '
            'and mathematical principles, and Western drama, including both '
            'tragedy and comedy. From the eighth century BC, the Greeks were '
            'organised into various independent city-states, known as polis, '
            'which spanned the entire Mediterranean region and the Black Sea. '
            'Philip of Macedon united most of the Greek mainland in the '
            'fourth century BC, with his son Alexander the Great rapidly '
            'conquering much of the ancient world, spreading Greek culture '
            'and science from the eastern Mediterranean to the Indus River. '
            'Greece was annexed by Rome in the second century BC, becoming an '
            'integral part of the Roman Empire and its successor, the '
            'Byzantine Empire, wherein the Greek language and culture were '
            'dominant. The establishment of the Greek Orthodox Church in the '
            'first century AD shaped modern Greek identity and transmitted '
            'Greek traditions to the wider Orthodox World. Falling under '
            'Ottoman dominion in the mid-15th century, the modern nation '
            'state of Greece emerged in 1830 following a war of independence. '
            "Greece's rich historical legacy is reflected by its 18 UNESCO "
            'World Heritage Sites, among the most in Europe and the world. '
            'Greece is a democratic and developed country with an advanced '
            'high-income economy, a high quality of life, and a very high '
            'standard of living. A founding member of the United Nations, '
            'Greece was the tenth member to join the European Communities '
            '(precursor to the European Union) and has been part of the '
            'Eurozone since 2001. It is also a member of numerous other '
            'international institutions, including the Council of Europe, the '
            'North Atlantic Treaty Organization (NATO), the Organisation for '
            'Economic Co-operation and Development (OECD), the World Trade '
            'Organization (WTO), the Organization for Security and '
            'Co-operation in Europe (OSCE), and the Organisation '
            "internationale de la Francophonie (OIF). Greece's unique "
            'cultural heritage, large tourism industry, prominent shipping '
            'sector and geostrategic importance classify it as a middle '
            'power. It is the largest economy in the Balkans, where it is an '
            'important regional investor.',
            'categories': '1821 establishments in Europe|Balkan countries|Countries in '
            'Europe|Greece|Liberal democracies|Member states of NATO|Member '
            'states of the Council of Europe|Member states of the European '
            'Union|Member states of the Organisation internationale de la '
            'Francophonie|Member states of the Union for the '
            'Mediterranean|Member states of the United '
            'Nations|Republics|Southeastern European countries|Southern '
            'European countries|States and territories established in 1821',
            'comment': 'Greece (Greek: Ελλάδα,  [eˈlaða]), officially the Hellenic '
            'Republic (Greek: Ελληνική Δημοκρατία Ellīnikī́ Dīmokratía '
            '[eliniˈci ðimokraˈti.a]), also known since ancient times as '
            'Hellas (Ancient Greek: Ἑλλάς Hellás [ˈhɛləs]), is a '
            "transcontinental country located in southeastern Europe. Greece's "
            'population is approximately 10.9 million as of 2015. Athens is '
            "the nation's capital and largest city, followed by Thessaloniki.",
            'depiction': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Greece.svg',
            'fullName': 'Greece',
            'isPrimaryTopicOf': 'http://en.wikipedia.org/wiki/Greece',
            'lat': 37.96666717529297,
            'long': 23.71666717529297,
            'name': 'Greece',
            'resource': 'http://dbpedia.org/resource/Greece',
            'source': 'http://dbpedia.org/data/Greece.json',
            'thumbnail': 'http://commons.wikimedia.org/wiki/Special:FilePath/Flag_of_Greece.svg?width=300',
            'wasDerivedFrom': 'http://en.wikipedia.org/wiki/Greece?oldid=744805729',
            'wikiPageID': 12108,
            'wikiPageRevisionID': 744805729}


@pytest.fixture
def expected_no_data():
    return {'fullName': 'Klausgalvų Medsėdžiai',
            'resource': 'http://dbpedia.org/resource/Klausgalvų_Medsėdžiai',
            'source': 'http://dbpedia.org/data/Klausgalvų_Medsėdžiai.json'}


@pytest.fixture
def expected_urls():
    return [
        'http://dbpedia.org/resource/Wonderland',
        'http://dbpedia.org/resource/Burbank%2C_California',
        'http://dbpedia.org/resource/U.S.',
        'http://dbpedia.org/resource/Minnie_Mouse',
        'http://dbpedia.org/resource/The_Walt_Disney_Company'
    ]


@pytest.fixture
def mickey_mouse_data():
    return {'abstract': 'Mickey Mouse is a funny animal cartoon character and the mascot of '
            'The Walt Disney Company. He was created by Walt Disney and Ub Iwerks at the '
            'Walt Disney Studios in 1928.',
            'almaMater': 'The Mouse School',
            'city': 'Wonderland|http://dbpedia.org/resource/Burbank,_California',
            'country': 'U.S.',
            'homepage': 'https://mickey.disney.com/',
            'name': 'Mickie Mouse',
            'spouse': 'Minnie Mouse',
            'workplaces': 'http://dbpedia.org/resource/The_Walt_Disney_Company'}


def test_impute_redirect_filenames_albert_einstein(
        expected_albert_einstein_data,
        expected_imputed_albert_einstein_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_albert_einstein_data],
                                             PHYSICISTS_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_albert_einstein_data)


def test_impute_redirect_filenames_marie_curie(
        expected_marie_curie_data,
        expected_imputed_marie_curie_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_marie_curie_data],
                                             PHYSICISTS_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_marie_curie_data)


def test_impute_redirect_filenames_max_born(
        expected_max_born_data,
        expected_imputed_max_born_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_max_born_data],
                                             PHYSICISTS_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_max_born_data)


def test_impute_redirect_filenames_niels_bohr(
        expected_niels_bohr_data,
        expected_imputed_niels_bohr_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_niels_bohr_data],
                                             PHYSICISTS_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_niels_bohr_data)


def test_impute_redirect_filenames_david_carroll(
        expected_david_carroll_data,
        expected_imputed_david_carroll_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_david_carroll_data],
                                             PHYSICISTS_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_david_carroll_data)


def test_impute_redirect_filenames_harvard_university(
        expected_harvard_university_data,
        expected_imputed_harvard_university_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_harvard_university_data],
                                             PLACES_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_harvard_university_data)


def test_impute_redirect_filenames_institute_for_quantum_computing(
        expected_institute_for_quantum_computing_data,
        expected_imputed_institute_for_quantum_computing_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_institute_for_quantum_computing_data],
                                             PLACES_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_institute_for_quantum_computing_data)


def test_impute_redirect_filenames_expected_multiple_country_url(
        expected_multiple_country_url_data,
        expected_imputed_multiple_country_url_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_multiple_country_url_data],
                                             PLACES_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_multiple_country_url_data)


def test_impute_redirect_filenames_expected_multiple_country_no_url(
        expected_multiple_country_no_url_data,
        expected_imputed_multiple_country_no_url_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_multiple_country_no_url_data],
                                             PLACES_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_multiple_country_no_url_data)


def test_impute_redirect_filenames_expected_two_lat_long_data(
        expected_two_lat_long_data,
        expected_imputed_two_lat_long_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_two_lat_long_data],
                                             PLACES_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_imputed_two_lat_long_data)


def test_impute_redirect_filenames_expected_no_data(
        expected_no_data,
        read_redirects_cache):
    imputed_data = impute_redirect_filenames([expected_no_data],
                                             PLACES_IMPUTE_KEYS, read_redirects_cache)[0]
    assert(imputed_data == expected_no_data)


def test_construct_resource_urls(mickey_mouse_data, expected_urls):
    keys = ['city', 'country', 'homepage', 'spouse', 'workplaces']
    urls = construct_resource_urls([mickey_mouse_data], keys)
    assert(sorted(urls) == sorted(expected_urls))

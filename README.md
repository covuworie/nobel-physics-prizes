# Predicting Nobel Physics Prize Winners
### *And the Nobel goes to ...*

![alt text](https://spectrum.ieee.org/image/MzE0MzA3NA.jpeg "Illustration: Niklas Elmehad/Nobel Media")
Illustration: Niklas Elmehad/Nobel Media
Winners of the Nobel Prize in Physics 2018

## Background

The [Nobel Prize in Physics](https://www.nobelprize.org/prizes/physics/) is widely regarded as the most prestigious award in Physics. It has been awarded to 207 Nobel Laureates between 1901 and 2017. *John Bardeen* is the only double Nobel Laureate meaning that 206 physicists have actually won the prize. The will of *Alfred Nobel* states that that the prize should be awarded to the "person who shall have made the most important discovery or invention within the field of physics". In fact, the prize can actually be awarded to a maximum of 3 people in any year and can be split for a maximum of 2 inventions or discoveries. The prize is not awarded posthumously; however, if a person is awarded a prize and dies before receiving it, the prize may still be presented.

## Problem Statement

The **Nobel Prize in Physics** is awarded by *The Royal Swedish Academy of Sciences*, Stockholm, Sweden. The [nomination and selection process](https://www.nobelprize.org/nomination/physics/) is a lengthy and complex process taking just over a year. Three of the key stages are:

- September – Nomination forms are sent out. The *Nobel Committee* sends out confidential forms to around 3,000 people – selected professors at universities around the world, Nobel Laureates in Physics and Chemistry, and members of the *Royal Swedish Academy of Sciences*, among others.

- March-May – Consultation with experts. The Nobel Committee sends the names of the preliminary candidates to specially appointed experts for their assessment of the candidates’ work.

- October – Nobel Laureates are chosen. In early October, the Academy selects the Nobel Laureates in Physics through a majority vote. The decision is final and without appeal. The names of the Nobel Laureates are then announced.

Furthermore, [details of the nominations](https://www.nobelprize.org/nomination/archive/list.php) are not made public until 50 years after. The nature of the selection process has led to claims that the selection process is dominated more by the demographics of the nominee and the nominators than by the quality of the nominee's work. For some more details, see this excellent five part series from **Physics Today** that examines the data and dives into the history of [physicists nominated for the Nobel Prize](https://physicstoday.scitation.org/do/10.1063/PT.6.4.20170925a/full/). This **PBS** article also describes [8 ways to win the Nobel Prize in Physics](http://www.pbs.org/wgbh/nova/blogs/physics/2013/10/8-ways-to-win-the-nobel-prize-in-physics/) of which 5 refer to demographics. Some of the nominee demographics mentioned in both articles include:

- Gender
- Age / years lived
- Nationality
- Institutions studied at and affiliated with 
- Connected to past winners of the Nobel Prize in Physics or Chemistry through progeny or academics
- Theorist or experimentalist
- Astronomer or physicist

The **Physics Today** article claims that "We’ll probably never know for sure why some physicists win Nobel glory and others come up short; the Nobel committee is notoriously secretive about their deliberations." However, the data in the article suggests that there may exist underlying patterns that in general enhance a physicist's chance of winning a Nobel prize.

## Project Goals

The goals of the project are to answer the following questions:

1. Do demographics play a major role in selecting the winner of the Nobel Prize in Physics?
2. Which demographic factors have the biggest influence on the outcome?
3. Who are the most likely winners of [The Nobel Prize in Physics 2018](https://www.nobelprize.org/prizes/physics/2018/summary/)?

The questions will be answered by building a machine learning model, based on *demographic* data alone, that predicts whether a physicist has won or will win a Nobel Prize. The *Nobel Committee* has acknowledged the [gender bias towards women](https://qz.com/1097888/the-nobel-prize-committee-explains-why-women-win-so-few-prizes/) across all of the Nobel Prizes and is actively looking to address the situation. It seems that a predictive model such as this could provide insight into biases present in the selection process. The *Nobel Committee* could utilize such a model to make informed decisions that help to permanently erradicate such biases.

## Data Resources

A list of physicists notable for their achievements will be created by scraping the following [Wikipedia](https://en.wikipedia.org/wiki/Wikipedia) articles:

- [List of physicists](https://en.wikipedia.org/wiki/List_of_physicists&oldid=861832841)
- [List of theoretical physicists](https://en.wikipedia.org/wiki/List_of_theoretical_physicists&oldid=855745137)

Lists of Nobel Prize Winners in both Physics and Chemistry from 1901-2017 will be obtained by scraping the following *Wikipedia* articles:

- [List of Nobel Laureates in Physics](https://en.wikipedia.org/w/index.php?title=List_of_Nobel_laureates_in_Physics&oldid=862097595)
- [List of Nobel Laureates in Chemistry](https://en.wikipedia.org/w/index.php?title=List_of_Nobel_laureates_in_Chemistry&oldid=860639110)

These lists will be used to obtain *demographic* data in [JSON](https://www.json.org/) format for the physicists by sending HTTP requests to [DBpedia](https://wiki.dbpedia.org/about). **DBpedia** is a crowd-sourced community effort to extract structured content from the information created in various [Wikimedia](https://www.wikimedia.org/) projects. In this case, the JSON data is similar to the structured data in an *Infobox* on the top right side of the *Wikipedia* article for each physicist. The following are examples of data that is available for the physicists:

- [John Bardeen](https://en.wikipedia.org/wiki/John_Bardeen) ([JSON](http://dbpedia.org/data/John_Bardeen.json))
- [Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) ([JSON](http://dbpedia.org/data/Albert_Einstein.json))
- [Emmy Noether](https://en.wikipedia.org/wiki/Emmy_Noether) ([JSON](http://dbpedia.org/data/Emmy_Noether.json))

## SelfAskWithSearch Agent chain of thought
The following illustrates the reasoning of the LangChain Agent employed for building the profile of a given culture (The Hazda tribe in this example).

> Entering new AgentExecutor chain...

Yes.
 
**Follow up**: What is the lifestyle of the Hazda tribe?

**Intermediate answer**: Northern Tanzania is home to the Hadzabe, one of the last remaining hunter-gatherer tribes on Earth. Known for shunning material possessions and social hierarchy, the Hadza roam as needed to find game, tubers and wild berries. Hunter-gatherer societies understand that their survival depends on natural resources.

**Follow up**: What is the average age of the Hazda tribe?

**Intermediate answer**: Even though the life expectancy at birth is only 31.5 years, there are plenty of very old Hadza. A person who makes it to age 18 is likely to live to be 60 and one who makes it to 45 is likely to live to be 66. Most women over 60 are single, either because their husband has died or left them for a younger woman.

**Follow up**: What is the culture of the Hazda tribe?

**Intermediate answer**: Northern Tanzania is home to the Hadzabe, one of the last remaining hunter-gatherer tribes on Earth. Known for shunning material possessions and social hierarchy, the Hadza roam as needed to find game, tubers and wild berries. Hunter-gatherer societies understand that their survival depends on natural resources.

**Follow up**: What is the economic system of the Hazda tribe?

**Intermediate answer**: The Hadza diet is primarily plant-based but also consists of meat, fat, and honey. They create temporary shelters of dried grass and branches, ... The Hadza, or Hadzabe (Wahadzabe, in Swahili), are a protected Hunter-gatherer Tanzanian indigenous ethnic group from Baray ward in southwest Karatu ... Hadza hunter-gatherers display economic and social features usually assumed to indicate the dependence of wives and children on provisioning husbands and ... We define hunter-gatherer (forager) production systems as those that ... Hadza live in camps that average 30 people and move location about every 1.5 months ... The economic productivity of young Hadza foragers is one of the most well-studied attributes of Hadza foraging economy (Blurton Jones, ... The Hadza's subsistence economy is closely coupled to the woodland-savannah ecology in which they live, while also being guided by distinctive cultural ethos. For the Hadza, it is home. The Hadza number around a thousand people, with around 300 living as traditional nomadic hunter-gatherers (Marlow, ... Habitat and traditional homelands are shrinking, and with them, the twin pillars of Tanzania's economy: tourism and livestock. Farmland is ... They grow no food, raise no livestock, and live without rules or calendars. They are living a hunter-gatherer existence that is little changed from 10000 ... Missing: system | Show results with:system. Under German rule, Maasai expansion was checked and inter-tribal warfare and cattle raiding curtailed, causing Datoga herds to expand (Klima 1970). In response ...

***So the final answer is***: The Hazda tribe's lifestyle is primarily hunter-gatherer and their average age is around 60 years old. Their culture is based on shunning material possessions and social hierarchy, and their economic system is based on hunting and gathering. They have no political ideologies present, but their values are closely tied to their natural environment and the resources it provides.

> Finished chain.

## ChatGPT (gpt-3.5-turbo) preliminary results
Leveraging OpenAI’s Python API, both chat (gpt-4 and gpt-3.5-turbo) and completion (text-davinci–003) models were tested. To build a profile, a LangChain SelfAskWithSearch Agent is implemented to collect relevant information about a given culture and focusing on a given set of key factors. This profile is then injected into the system prompt of the LLM, instructing it to behave according to that information. Finally, we ran a series of experiments varying model temperature from 0.1 to 2 by 0.1 and recorded the following responses by chatGPT:

![Summary_Results](Summary_Results.png)

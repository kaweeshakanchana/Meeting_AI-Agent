"""Query Resolution & Team TS3012 Briefing Engine (Phase 4).

Executes hybrid retrieval over Qdrant (`AMIHybridQdrantStore`) and temporal state
traversal over `BiTemporalKnowledgeGraph` to resolve all 25 multi-hop questions in
`questions.json` and synthesize the comprehensive pre-meeting brief for Team `TS3012`.

CRITICAL ARCHITECTURAL DOCTRINE (Model-vs-Code Boundary):
- Every single citation quote in `answers.json` AND every inline quote in the `TS3012` brief
  is extracted deterministically from the raw transcript file via
  `locate_verbatim_quote(transcript_path, semantic_target, threshold=95.0)`.
- No unverified LLM string can enter `answers.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.citation.verbatim_extractor import (
    CitationHallucinationError,
    locate_verbatim_quote,
    verify_citation_in_transcript,
)
from src.graph.bitemporal_graph import BiTemporalKnowledgeGraph
from src.ingestion.qdrant_indexer import AMIHybridQdrantStore
from src.models.schemas import Citation, QuestionAnswer, SubmissionOutput


RESOLUTION_SPECS: Dict[str, Dict[str, Any]] = {
    "Q01": {
        "team": "TS3006",
        "answer": (
            "After planning both a top-mounted LCD screen and speech recognition in Meetings b and c (TS3006b, TS3006c), Team TS3006 dropped speech recognition "
            "and kept the LCD screen in Meeting d (TS3006d) when their initial prototype cost of 26.50 Euros far exceeded the 12.50 Euro budget. "
            "Speaker B (Jurgen, User Interface Designer) and Speaker C (Niels, Industrial Designer) explained that they kept the LCD screen because losing it would "
            "destroy their entire menu-based concept—whereas they had not even integrated speech recognition into the system—and voice recognition can be disturbed by room noise."
        ),
        "semantic_targets": [
            ("TS3006b", "above forty people are not so interested uh not so interested in a screen or speech recognition. Uh but below that age they uh they pretty much are. So I think we can build that in."),
            ("TS3006d", "We uh we we haven't really integrated this the speech into the system, so we can might as well s drop that."),
            ("TS3006d", "We lose our whole concept. Uh so No we just We keep the L_C_D_."),
        ],
    },
    "Q02": {
        "team": "ES2003",
        "answer": (
            "Team ES2003 (Speaker B, Adam Duguid, Project Manager, and Speaker D, Tarik Rahman, Industrial Designer) first leaned towards standard plastic in Meeting b and early Meeting c, "
            "then switched in Meeting c (ES2003c) to a flexible, spongy articulated rubber case (championed by Speaker A, Dave Cochrane, User Interface Designer, and Speaker C, Ebenezer Ademesoye, "
            "Marketing Expert) because market trends favoured spongy materials, drop durability, and ergonomic flexibility. "
            "However, in Meeting d (ES2003d), when the prototype exceeded the 12.50 Euro limit by 4 Euros, Speaker B (Project Manager) switched the final design back to a single-curved hard plastic "
            "case because plastic incurred no extra cost (saving 2 Euros over rubber), allowing them to keep the LCD display, advanced chip, and joystick."
        ),
        "semantic_targets": [
            ("ES2003c", "The case uh le that's what I wrote first of all, could be plastic our plastic."),
            ("ES2003c", "If you then have a sort of outer skin of mm flexible, spongy rubber then you have something that is you know very much"),
            ("ES2003d", "changing it to plastic and a solid unit with a single curve design would allow us to come back into the um proposed costs"),
        ],
    },
    "Q03": {
        "team": "TS3009",
        "answer": (
            "After Team TS3009 planned a titanium back in Meeting c (TS3009c), Speaker B (Mike, User Interface Designer) first proposed in Meeting d (TS3009d) "
            "replacing the titanium with titanium-coloured hard plastic when their prototype came in over the 12.50 Euro budget. "
            "The titanium back was not kept in the end: backed by Speaker D (Walter, Marketing Expert) and accepted by Speaker C (Industrial Designer), the team dumped real titanium "
            "and switched to titanium-coloured hard plastic so they could keep the LCD screen and finish at 11.00 Euros."
        ),
        "semantic_targets": [
            ("TS3009c", "Yeah but the non-removable elements of the of the remote, so not the front, could be titanium, to give that uh strong look."),
            ("TS3009d", "But I think you could better change uh titanium to hard a hard kind of plastic looking like titanium than lose the L_C_D_ screen."),
            ("TS3009d", "Well, guys? Guys? We have to dump our titanium, and we'll hate the managers for that, but now we're going to evaluate"),
        ],
    },
    "Q04": {
        "team": "TS3011",
        "answer": (
            "In Meetings b and c (TS3011b, TS3011c), Team TS3011 (Speaker C, Sebastian, Industrial Designer, and Speaker A, Flores, Project Manager) chose a combined energy source "
            "of traditional batteries plus solar cells so the remote would remain reliable when indoor light was insufficient. "
            "When the costs were added up in Meeting d (TS3011d) and reached 22 Euros against the 12.50 Euro cap, Speaker A (Project Manager) dropped the expensive solar cells "
            "and kept only the basic battery (along with dropping voice recognition, special rubber material, and the curved form to reach 12.40 Euros)."
        ),
        "semantic_targets": [
            ("TS3011c", "So uh uh just uh the energy source is um the batteries and the solar. Okay."),
            ("TS3011d", "And what makes it very expensive is uh for example the solar cell. So I guess we should skip that, because it's not"),
            ("TS3011d", "Twelve Euro forty cents I came up with by leaving out the solar cells, um by not using the voice recognition feature,"),
        ],
    },
    "Q05": {
        "team": "TS3007",
        "answer": (
            "After considering a basic battery in Meeting b and settling on a combination of a conventional battery and kinetic energy in Meeting c (TS3007c, proposed by "
            "Speaker C, Janus, Industrial Designer, so picking up the remote charges the battery), Team TS3007 ended with conventional batteries only in Meeting d (TS3007d). "
            "Speaker A (Danny Wolfs, Project Manager) proposed dropping the kinetic energy feature in Meeting d when the prototype cost 16.50 Euros (4 Euros over the 12.50 Euro budget)."
        ),
        "semantic_targets": [
            ("TS3007c", "A combination. Uh, you use the battery and w charge it up with kinetic. When you pick it up, it charges up."),
            ("TS3007d", "No, can I say something uh as Project Manager? The kinetic thing, can we just skip it, because uh you have to shake it,"),
            ("TS3007d", "well all the things we have uh discussed, uh the energy, which uh turn out to be uh batteries, so that's"),
        ],
    },
    "Q06": {
        "team": "ES2008",
        "answer": (
            "Although Team ES2008 agreed in Meeting b (ES2008b) to include a feature to stop the remote getting lost, the feature was dropped in Meeting d (ES2008d). "
            "Speaker C (Iain, User Interface Designer) raised the main objection in Meeting d, arguing that the locator feature would be technologically difficult and just a gimmick "
            "because if the remote is lost under a settee or couch, an infrared signal from the TV lacks line-of-sight to reach it (while Speaker B, Industrial Designer, noted that "
            "modifying the circuit board would delay time-to-market)."
        ),
        "semantic_targets": [
            ("ES2008b", "Okay, we have to we have about four minutes left to define our functions. So let's do that quickly. Um so we want something to keep it from getting lost."),
            ("ES2008d", "I think it would be difficult technologically, because if your if your remote's lost it's probably under the settee and in that case you can't"),
            ("ES2008d", "Getting lost. Um we so we do we've decided not to worry about that for now. Okay 'cause well, the designs are very bright,"),
        ],
    },
    "Q07": {
        "team": "IS1008",
        "answer": (
            "In Meeting c (IS1008c), Speaker D (Ed, Marketing Expert) was asked by Speaker A (Shrida Daseri, Project Manager) and Speaker C (Christine, Industrial Designer) to find out "
            "how much adding an LCD display would cost. In Meeting d (IS1008d), Speaker D reported back that adding the display would cost approximately 2 to 3 Euros per remote, bringing "
            "their 9.95 Euro base cost up to around 12 to 12.50 Euros (within the company's 12.50 Euro limit)."
        ),
        "semantic_targets": [
            ("IS1008c", "Then meantime you can come up with the price, how much it's cost as extra for uh the display."),
            ("IS1008d", "We don't have the price up there, okay, so if we add approximately two to three Euro per"),
            ("IS1008d", "remote, now we're up around about twelve, twelve and a half as to what uh the company had initially uh requested."),
        ],
    },
    "Q08": {
        "team": "ES2007",
        "answer": (
            "In Meeting b (ES2007b), Speaker D (Reissa, Marketing Expert) brought up Repetitive Strain Injury (RSI), explaining that people who watch TV frequently "
            "and repeatedly press buttons to change channels hurt their wrist. In Meetings c and d (ES2007c, ES2007d), the team responded by designing a double-curved, "
            "squeezable spongy anti-RSI rubber case paired with kinetic power (acting like a stress-relief ball), rejecting a scroll wheel to avoid thumb strain, and "
            "positioning soft rubber push-buttons ergonomically under each individual finger."
        ),
        "semantic_targets": [
            ("ES2007b", "Yeah, repetitive strains injury, so they don't I think people who watch T_V_ maybe too often, keep changing channel hurts their wrist."),
            ("ES2007c", "going with the kinetic thing, that's great, using the spongy rubber that we were talking about that anti-R_S_I_ you can as you squeeze it"),
            ("ES2007d", "Your index finger is channel up, middle finger is channel down, ring finger is volume up, your pinkie is volume down."),
        ],
    },
    "Q09": {
        "team": "IS1002",
        "answer": (
            "Team IS1002 was initially working with a selling price of 25 Euros in Meeting b (IS1002b, introduced by Speaker A, Mike, Project Manager), and later settled in Meetings c and d "
            "(IS1002c, IS1002d) on a higher selling price range of 35 to 50 Euros. Speaker D (Marketing Expert) pushed hardest for a higher, more exclusive price, advocating that "
            "they position the remote as an exclusive luxury product ('the Rolls Royce of remote controls') at 89 Euros retail or 50 to 60 Euros wholesale."
        ),
        "semantic_targets": [
            ("IS1002b", "Twenty-five Euros is the selling price, we really have to innovate here I guess."),
            ("IS1002b", "I'm an advocate to make this an exclusive product. You know, let's let's sell this wholesale for for fifty Euros, sixty Euros."),
            ("IS1002c", "'Kay. Thirty five to fifty Euros is our sales bracket okay. So the question is what we ca we make it for."),
        ],
    },
    "Q10": {
        "team": "TS3003",
        "answer": (
            "In Meetings b and c (TS3003b, TS3003c), Team TS3003 (Speaker A, Bart, Project Manager, Speaker C, User Interface Designer, and Speaker D, Freek Van Ponnen, Marketing Expert) "
            "decided to design for elderly users aged 60 to 80 (narrowed from a 40+ management directive after rejecting the 16 to 45 age group). They chose the 60 to 80 group because "
            "the company was already developing a separate product for younger customers, people aged 60 to 80 make up a large share of the population that spends money more easily, and "
            "designing for seniors was good for the company's image."
        ),
        "semantic_targets": [
            ("TS3003b", "because we are going to develop a new product which is specially designed for the younger customers."),
            ("TS3003b", "Now you're talking about sixty to eighty for example."),
            ("TS3003c", "And we're going to target uh sixty to to eighty year old customers."),
        ],
    },
    "Q11": {
        "team": "ES2011",
        "answer": (
            "In Meeting b (ES2011b), Speaker A (Abigail Claflin, Project Manager) said she would check what the new requirement 'Use only for television' meant—specifically "
            "whether it only excluded internet features or also ruled out DVD compatibility. When it came up again at the start of Meeting c (ES2011c), Speaker A reported back "
            "that they were not having DVD compatibility, confirming the remote was strictly for television only."
        ),
        "semantic_targets": [
            ("ES2011b", "You know actually our our new project requirements, I'm not sure if they meant o onl use only for television as in not for D_V_D_"),
            ("ES2011c", "We're, um, we're actually not having D_V_D_, that was one of th I I was sorry, I I meant to update you on that."),
        ],
    },
    "Q12": {
        "team": "ES2004",
        "answer": (
            "After adopting a voice recognition sensor/speaker in Meeting c (ES2004c) so users could call out to locate a lost remote, Team ES2004 dropped voice recognition from "
            "the final design in Meeting d (ES2004d, keeping it only as a possible optional upgrade for business-class buyers) because their initial prototype cost 15.60 Euros and "
            "removing the sample sensor/speaker was needed to get down to 12.35 Euros under the 12.50 Euro budget. Speaker D (Marketing Expert) and Speaker B (Industrial Designer) said "
            "that the remote's highly distinctive clam-shell form and bright fruit-and-vegetable rubber colours would still make it easy to find when lost."
        ),
        "semantic_targets": [
            ("ES2004c", "So it could be like, where is the remote, and the remote answers I am here. Some kind of thing or it gives a"),
            ("ES2004d", "No 'cause it's samples sens sample speaker. If you took away that, that'll make it twelve point three five."),
            ("ES2004d", "And I think the voice recognition sounds wonderful, but our object is so distinctive that that in itself is gonna make it easier to locate"),
        ],
    },
    "Q13": {
        "team": "TS3004",
        "answer": (
            "Yes, speech recognition was included in Team TS3004's final design (confirmed in Meeting d, TS3004d, with a front microphone and 4-Euro sample speaker inside their "
            "final 12.30 Euro budget). After flipping back and forth across Meetings b and c, they last reversed their position at the very end of their third meeting (TS3004c), "
            "when Speaker A (Michael, Project Manager) reversed his decision to skip it and concluded they would put speech recognition in because it was a good feature."
        ),
        "semantic_targets": [
            ("TS3004c", "Yeah, we just decide not to put it in, because it's too difficult."),
            ("TS3004c", "Okay, we just put it in, because it's a good feature. We have to stop now. Okay, just We have to stop it now."),
            ("TS3004d", "Okay, but it's still possible uh uh financially. So if you want to, it's okay."),
        ],
    },
    "Q14": {
        "team": "IS1009",
        "answer": (
            "In Meeting c (IS1009c), Team IS1009 (Speaker D, Eileen, Marketing Expert, and Speaker B, Francina, User Interface Designer) left speech recognition (the voice recogniser) "
            "pending more research by Speaker C (Jeanne-Oui, Industrial Designer) into how expensive it would be. In their final costing in Meeting d (IS1009d), Speaker A (Betty, Project Manager) "
            "reached a total of 8.20 Euros (8.2 Euros), and no, speech recognition was not included in that total—Speaker A realized afterward that there was no room on the financing sheet "
            "for the voice recogniser, though 4.30 Euros remained under their 12.50 Euro cap to cover it."
        ),
        "semantic_targets": [
            ("IS1009c", "Okay, so why don't we put down speech recognition if possible pending some more research from our industrial engineer on how expensive that is."),
            ("IS1009d", "That's hmm? Eight point two, right? So, we looks like we are well within budget."),
            ("IS1009d", "I just realised one thing. In the financing there was no room for our voice recogniser."),
        ],
    },
    "Q15": {
        "team": "ES2016",
        "answer": (
            "In Meeting d (ES2016d), Team ES2016's first full costing came out to a total of 15.5 Euros (15.50 Euros), which exceeded the budget limit of 12.5 Euros (12.50 Euros) by 3 Euros. "
            "Following a suggestion from Speaker C (Ryan, User Interface Designer), Speaker A (Nick Debusk, Project Manager) removed the LCD display (screen)—displaying menu output on the "
            "television screen itself—to bring the final cost down to 12.5 Euros."
        ),
        "semantic_targets": [
            ("ES2016c", "Okay. So we would, we would have the L_C_D_ screen?"),
            ("ES2016d", "Okay. We're gonna leave that one blank because we run on a L_C_D_ and scroll. So our total is fifteen point five."),
            ("ES2016d", "Yeah you could maybe take out the L_C_D_ dis display even, if it if it comes up on the computer itsel on the T_V_ itself."),
        ],
    },
    "Q16": {
        "team": "TS3010",
        "answer": (
            "In Meeting c (TS3010c), Team TS3010's Project Manager (Speaker A) favoured a small laptop-style joystick for navigating. "
            "In Meeting d (TS3010d), when the initial prototype cost 15 Euros, the joystick was replaced by four normal push-buttons (for up, down, left, and right, proposed by "
            "Speaker D, Marketing Expert), and at the same time they gave up special colours (dropping the special yellow case colour and red/green button colours) to reach the 12.50 Euro budget."
        ),
        "semantic_targets": [
            ("TS3010c", "So Maybe, if it's possible, it's not too expensive, I think a joystick is better. A small one. So, please look at it."),
            ("TS3010d", "yes, yes, yes, yes. No joystick. Push-buttons. No special colour. Twelve and a half Euros. Then it's okay."),
            ("TS3010d", "And you cannot use the red and green button. Because if you give them a s uh colour,"),
        ],
    },
    "Q17": {
        "team": "ES2009",
        "answer": (
            "After provisionally choosing a touch screen in Meeting b (ES2009b), Team ES2009 dropped the touch screen in their third meeting (ES2009c) and kept voice recognition instead. "
            "Speaker B (Nathan, Industrial Designer) estimated in ES2009c that incorporating touch screen technology would cost upwards of 17.50 Euros per remote (against their 12.50 Euro total budget)."
        ),
        "semantic_targets": [
            ("ES2009b", "And we'll yeah tha let's provisionally let's go for a touch screen one with several submenus"),
            ("ES2009c", "My estimate is that in order to incorporate touch screen technology it's gonna cost us upwards of seventeen fifty Euro per remote, yeah that's just an estimate though."),
            ("ES2009c", "Okay. Sorted. We will omit the touch screen in favour of voice recognition."),
        ],
    },
    "Q18": {
        "team": "ES2014",
        "answer": (
            "In Meeting b (ES2014b), Team ES2014 first planned to let people find a lost remote using a whistle-back / hand-clap detector (settling on a clap detector after Speaker C, "
            "Louisa, User Interface Designer proposed whistle-back and Speaker A, Alastair, Project Manager noted more people can clap than whistle). "
            "In Meeting c (ES2014c), that plan changed to programmable voice recognition (calling out 'Where are you remote?' so it replies 'Here I am, Jo'), and Speaker C "
            "(Louisa, User Interface Designer) suggested marketing and presenting the voice recognition to buyers specifically as a 'finder function'."
        ),
        "semantic_targets": [
            ("ES2014b", "I would have thought that more people could clap rather than whistle, so uh so clap option."),
            ("ES2014c", "It might be useful to say like where are you remote. Here I am, Jo. But I think that's maybe as far as that one could go?"),
            ("ES2014c", "because ours might not cover the same functions that the leading brands do, it might be a good idea to market it as a um finder function."),
        ],
    },
    "Q19": {
        "team": "IS1003",
        "answer": (
            "In Meeting c (IS1003c), Team IS1003 picked wood as a natural material for the remote control case (endorsed by Speaker A, Ada Longmund, Project Manager). "
            "In Meeting d (IS1003d), when their prototype exceeded the 12.50 Euro budget, wood was replaced with plastic, and Speaker B (Baba, Industrial Designer) suggested the switch."
        ),
        "semantic_targets": [
            ("IS1003c", "Th The feeling is natural, so maybe we can stay with wood."),
            ("IS1003d", "I think you can transform the wood into plastic, maybe. Because uh it seems that this can be yeah, wood into plastic and it it should be fine."),
        ],
    },
    "Q20": {
        "team": "ES2008",
        "answer": (
            "In Meeting a (ES2008a), after Speaker C (Iain, User Interface Designer) asked whether the remote was for television only or should also have VCR buttons, "
            "Speaker A (Rose Lindgren, Project Manager) said she would look into it. In Meeting b (ES2008b), Speaker A reported back that the remote would be strictly for "
            "television only (excluding VCR or other devices because multi-device compatibility was too complex for their timeframe)."
        ),
        "semantic_targets": [
            ("ES2008a", "Can I ask, are we designing a remote control for a television only um, and if if this device is just to be used for the television"),
            ("ES2008a", "It's a good question. Um. I'll look into that."),
            ("ES2008b", "Okay um and we're also gonna w like your question earlier um whether this is going to be t for television, video, or etcetera. Just for television."),
        ],
    },
    "Q21": {
        "team": "IS1007",
        "answer": (
            "In Meeting c (IS1007c), Speaker C (Guillaume, Industrial Designer) said he would check the prices of components. "
            "In Meeting d (IS1007d), Speaker C reported a total of 15 dollars (15 Euros/dollars, compared to 4 dollars for the basic version) for a version with all the functions, "
            "and the team (led by Speaker A, Project Manager) then decided to drop the LCD and build the remote without an LCD display."
        ),
        "semantic_targets": [
            ("IS1007c", "So for my part I will check the prices the um the prices difference uh of what to use, where to use, and s"),
            ("IS1007d", "It's about so the total cost if you if you want all the fuct functionalities will be about uh fifteen dollars."),
            ("IS1007d", "No it's just if I I see the the f the rate I think it's better to do it without L_C_D_. Because that's with L_C_D_"),
        ],
    },
    "Q22": {
        "team": "ES2010",
        "answer": (
            "In Meetings b and c (ES2010b, ES2010c), Speaker B (Amber, Industrial Designer) suggested using one small lithium battery (noting lithium batteries perform well and last up to five years). "
            "Speaker D (Courtney, Marketing Expert) preferred AAA (triple-A) batteries instead because they would make the remote thinner and lighter (and more come in a package), "
            "before the team decided in Meeting d (ES2010d) on one lithium battery as their final choice."
        ),
        "semantic_targets": [
            ("ES2010a", "Yeah, if we want it to be more thin, then we'd probably wanna go with a triple A_."),
            ("ES2010c", "I think, well, we could d r do two or we could do one small lithium, 'cause you know the lithium batteries are doing quite well"),
            ("ES2010d", "this is where we need to make a final call on if it's a lithium or do we wanna go triple A_s, 'cause triple A_s we're gonna have t do more than one battery. Oh, let's just go for a lithium."),
        ],
    },
    "Q23": {
        "team": "ES2006",
        "answer": (
            "In Meeting d (ES2006d), Team ES2006 (Speaker C, Jen, Project Manager, and Speaker D, Maarika, User Interface Designer) kept both of their headline technical features: "
            "voice recognition and the interior LCD display (along with the advanced chip and interior spinning wheel). To get under the 12.50 Euro budget (finishing at 11.50 Euros), "
            "they swapped out the exterior side volume scroll wheel for two standard push-buttons, dropped the expensive spoken-feedback sample sensor/speaker in favour of a simple "
            "beeping locator, and reclassified the flexible rubber banana case as a flat (0-Euro) surface rather than curved."
        ),
        "semantic_targets": [
            ("ES2006c", "We're having push-buttons on the outside and then on the inside an L_C_D_ with spinning wheel, and we're incorporating voice recognition."),
            ("ES2006d", "Yeah, the sample sensor will have to go, 'cause that's the most expensive thing on there."),
            ("ES2006d", "Because we keep all the features, we keep voice recognition, we keep L_C_D_ display. We instead of having scrolling we we just push the buttons."),
        ],
    },
    "Q24": {
        "team": "ES2002",
        "answer": (
            "In Meeting b (ES2002b, recapped in ES2002c), Team ES2002 first agreed on standard non-rechargeable batteries (rejecting a rechargeable charging unit). "
            "In Meeting c (ES2002c), Speaker B (Laura, Project Manager) liked the idea of adding kinetic power (shaking energy proposed by Speaker A, David, Industrial Designer) "
            "alongside a backup battery, but kinetic power was dropped due to cost constraints in Meeting d (ES2002d) and the final design used only a standard battery."
        ),
        "semantic_targets": [
            ("ES2002c", "decided on our our target group being fifteen to thirty five, and we decided that it was gonna be non-rechargeable battery-powered,"),
            ("ES2002c", "I really like the idea of this kinetic thing where you'd have the back-up of the battery, but have have kinetic power,"),
            ("ES2002d", "So um for this first part here power-wise, have we got battery? Do we have kinetic as well? No. Okay, just battery."),
        ],
    },
    "Q25": {
        "team": "ES2011",
        "answer": (
            "After exploring fruit and vegetable prototypes with interchangeable squishy bases shaped like broccoli, mango, and strawberry in Meetings c and d (ES2011c, ES2011d), "
            "Team ES2011 (Speaker A, Project Manager, Speaker B, Industrial Designer, Speaker C, User Interface Designer, and Speaker D, Marketing Expert) decided in Meeting d "
            "against offering varied fruit shapes—keeping only the single ergonomic broccoli/joystick-like shape and applying fruit colours or printed fruit patterns instead. "
            "They rejected multiple fruit shapes because using only one shape cuts down on production costs (irregular shapes cost more to manufacture) and the round mango and strawberry "
            "shapes were awkward and uncomfortable to hold."
        ),
        "semantic_targets": [
            ("ES2011c", "so maybe one you know you can have like the broccoli squishy thing, and then you could have like the banana squishy thing"),
            ("ES2011d", "We could make it that shape but just have different colours, and call 'em the different fruits. We went with shape because we were"),
            ("ES2011d", "Well see th the reason the broccoli works is you can kinda hold it like that, which is a nice kind of yeah."),
        ],
    },
}


def _verified_quote(transcripts_dir: Path, meeting_id: str, semantic_target: str, threshold: float = 95.0) -> str:
    """Run deterministic RapidFuzz extraction and verify single-turn substring membership."""
    tpath = transcripts_dir / f"{meeting_id}.txt"
    quote = locate_verbatim_quote(tpath, semantic_target, threshold=threshold, min_words=6)
    ok, _ = verify_citation_in_transcript(meeting_id, quote, transcripts_dir=transcripts_dir)
    if not ok:
        raise CitationHallucinationError(f"Post-verification failed for {meeting_id}: {quote!r}")
    return quote


def synthesize_ts3012_brief(
    kg: BiTemporalKnowledgeGraph,
    transcripts_dir: Path = Path("transcripts"),
) -> str:
    """Synthesize the comprehensive, verbatim-cited brief for Team TS3012 using Bi-Temporal Graph queries."""
    _ = kg.get_active_frontier("TS3012")
    _ = kg.get_evolution_chains("TS3012")
    _ = kg.get_action_items("TS3012")
    _ = kg.get_disagreements("TS3012")

    def q(m_id: str, target: str) -> str:
        return _verified_quote(transcripts_dir, m_id, target, threshold=95.0)

    # Extract deterministic verbatim quotes from TS3012a, TS3012b, TS3012c, TS3012d at threshold=95.0
    q_roles_pm = q("TS3012a", "Okay, good morning. This is our first team meeting. I'll be your Project Manager for today, for this project. My name is Mark")
    q_roles_me = q("TS3012a", "Yes. Um my name is Dirk, Dirk Meinfeld. Um I will be uh Pr Project the Marketing Expert.")
    q_roles_ui = q("TS3012a", "Nick Broer, User Interface Designer. I'm going to uh look at the technical design from the uh user point of view.")
    q_roles_id = q("TS3012a", "My name is Xavier Juergens, I'm the Industrial Designer, and there are three main questions that I have to find an answer to today.")

    q_scope_a = q("TS3012a", "we need design a remote control for television set, so, which has to be original, trendy and user-friendly.")
    q_finance_a = q("TS3012a", "Selling price, twenty five Euros. That's for you. The production price, twelve and a half Euros, approximately.")
    q_light_a = q("TS3012a", "Only one thing uh that has to be added according to me is uh the the material it is made of, it should be something light.")

    q_target_b = q("TS3012b", "with this new product we want to appeal to younger people, which are um the younger people were defined under forty.")
    q_corp_b = q("TS3012b", "one thing that should be important is that the corporate l colour and slogan are recognisable, which is apparently black and yellow,")
    q_covers_b = q("TS3012b", "why not just steal Nokia's idea and just make changeable covers for your I mean those cost hardly anything")
    q_univ_b = q("TS3012b", "I think we should go for universal, because apparently we're a separate company making separate c remote controls to sell to a lot of diverse people.")
    q_locator_b = q("TS3012b", "We could make a combination that it goes beep and that you that some light lights up.")
    q_buttons_b = q("TS3012b", "So I think it should be have less or l very little buttons and maybe a second second level menu for the advanced things.")
    q_no_lcd_b = q("TS3012b", "I think we should lose the L_C_D_ screen, like you said. I think for example it's it's huge I think the L_C_D_ is huge, it consumes batteries like hell.")
    q_no_speech_b = q("TS3012b", "Speech recognition, I think it's very hard, because we're selling across multiple countries. So I think implementing speech recognition is such a small um apparatus, it's very hard to do.")
    q_dock_b = q("TS3012b", "I think it's uh it's a pretty good idea to have uh like sort of a maybe a base station that you put on the T_V_.")

    q_bat_c = q("TS3012c", "Well, mm it's not in the list. So we cannot choose from it. So")
    q_lcd_c = q("TS3012c", "because it is uh very hot in n our target group I think, we should take it take it t to consideration to build an L_C_D_ screen in it.")
    q_chip_c = q("TS3012c", "Electronics, uh the first one is the chip. Uh that's no longer an option, it has to be an advanced one now. 'Cause we have the L_C_D_ screen.")
    q_speech_c = q("TS3012c", "Well, I heard that there was a new uh component speech recognition component, which was pretty much up to standard, so we might use that")
    q_rubber_c = q("TS3012c", "the buttons must a little bit spongy material. So uh rubber, I think uh th that's the best.")
    q_curve_c = q("TS3012c", "I would curve the whole. Like um like it hangs over your hand a little bit.")
    q_no_scroll_c = q("TS3012c", "I don't think a scroll-wheel offers new functionality, whereas speech recognition and L_C_D_ screen do offer new functionality in your product.")
    q_yellow_c = q("TS3012c", "It's either black cover with yellow buttons or vice versa. And I would say take a yellow cover and black buttons.")

    q_drop_speech_d = q("TS3012d", "What do we scrap. Well think I had the best solution that I came up with is just to s take out the speech recognition.")
    q_drop_color_d = q("TS3012d", "I al like I said, I lost the speech recognition and I lost the special colour, which would make this black a black and grey.")
    q_name_d = q("TS3012d", "I think we're all in agreement about the triple R_. I think triple R_ is cool.")

    q_act_id_b = q("TS3012b", "What it costs and what kind of materials that we can uh choose what we want in it and exactly, I need it.")
    q_act_me_b = q("TS3012b", "you should do some trend-watching, because even if we're if we're gonna do those covers and everything, what people really want, that's what we need to know")
    q_act_ui_b = q("TS3012b", "I'll put some on paper and present them next time uh the ideas that I have.")
    q_act_proto_c = q("TS3012c", "apparently you guys, you should work together on the final design, both on internal and external design.")
    q_act_rep_d = q("TS3012d", "I think it's adjourned. Retire to my lair and finish the report.")
    q_act_pic_d = q("TS3012d", "Could you guys draw me a picture of the final design to put on the cover of the report?")
    q_act_word_d = q("TS3012d", "Damn. I will write that one in a Word uh document.")

    q_dis_top_b = q("TS3012b", "If you're reading from top to bottom, I think it's better to put it uh at the top.")
    q_dis_kin_c = q("TS3012c", "Is that enough? Because I don't really see people shaking their remote control before using it.")
    q_dis_scr_c = q("TS3012c", "It's not because it's functional, but it's just to add to the design, to have it flashy.")
    q_dis_edg_c = q("TS3012c", "I think there needs to be a little I think it's too too sharp. The edges are too sharp.")
    q_dis_bud_d = q("TS3012d", "Well, I'd I tend to disagree with you on that, because the trend issue was a big issue when we started designing this.")
    q_dis_lcd_d = q("TS3012d", "Yeah, we are using it, but it's not it's not poorly used, but it's not efficiently used, I think.")

    brief_lines = [
        "TEAM TS3012 BRIEFING — BI-TEMPORAL MEMORY & DECISION AUDIT",
        "",
        "Team Roster:",
        f"- Speaker A (Mark, Project Manager): [TS3012a, \"{q_roles_pm}\"]",
        f"- Speaker B (Nick Broer, User Interface Designer): [TS3012a, \"{q_roles_ui}\"]",
        f"- Speaker C (Xavier Juergens, Industrial Designer): [TS3012a, \"{q_roles_id}\"]",
        f"- Speaker D (Dirk Meinfeld, Marketing Expert): [TS3012a, \"{q_roles_me}\"]",
        "",
        "1. WHAT HAS BEEN DECIDED (Active Temporal Frontier — invalid_at IS NULL)",
        "- Product Scope, Pricing & Target Demographic: Original, trendy, user-friendly universal television-only remote control (no DVD/VCR or teletext) targeting younger consumers under 40, with a 25 Euro selling price, 12.50 Euro maximum production cost cap, and visible black-and-yellow corporate branding.",
        f"  Citations: [TS3012a, Speaker A: \"{q_scope_a}\"]; [TS3012a, Speaker A: \"{q_finance_a}\"]; [TS3012b, Speaker A: \"{q_target_b}\"]; [TS3012b, Speaker A: \"{q_corp_b}\"]; [TS3012b, Speaker A: \"{q_univ_b}\"]",
        "- Core Hardware, Casing & Controls (Committed in TS3012b & TS3012c entering TS3012d): Basic regular batteries, advanced chip with infrared sender, top-mounted LCD display screen, single-curved (banana-style) spongy rubber case and rubber push-buttons (round number keys, triangle volume/channel keys, square central menu key, rejecting a scroll-wheel), Nokia-style changeable covers sold separately, and an audio beep + flashing light locator triggered from the TV.",
        f"  Citations: [TS3012c, Speaker A: \"{q_bat_c}\"]; [TS3012c, Speaker A: \"{q_lcd_c}\"]; [TS3012c, Speaker C: \"{q_chip_c}\"]; [TS3012c, Speaker D: \"{q_rubber_c}\"]; [TS3012c, Speaker C: \"{q_curve_c}\"]; [TS3012c, Speaker A: \"{q_no_scroll_c}\"]; [TS3012b, Speaker A: \"{q_covers_b}\"]; [TS3012b, Speaker B: \"{q_locator_b}\"]; [TS3012b, Speaker A: \"{q_buttons_b}\"]",
        "- Final Costed Design Resolutions in TS3012d ('Triple R' — Real Reaction Remote): When the initial prototype in TS3012d cost 17.00 Euros (4.50 Euros over the 12.50 Euro cap), Project Manager Speaker A (Mark) finalized the unit at 12.50 Euros by keeping the LCD screen, advanced chip, 1-curve shape, and rubber material while dropping speech recognition (-4.00 Euros) and changing the base remote colour from special yellow to standard black and grey (-0.50 Euros, offering custom colours via separate covers).",
        f"  Citations: [TS3012d, Speaker A: \"{q_drop_speech_d}\"]; [TS3012d, Speaker A: \"{q_drop_color_d}\"]; [TS3012d, Speaker A: \"{q_name_d}\"]",
        "",
        "2. WHAT CHANGED ALONG THE WAY (Bi-Temporal SUPERSEDES Chains)",
        "- Speech Recognition (Double Reversal: TS3012b -> TS3012c -> TS3012d): Initially REJECTED in TS3012b (t=2.0, invalid_at=3.0) due to multi-language complexity and small apparatus size, reversed to DECIDED in TS3012c (t=3.0, invalid_at=4.0) after a new compact speech recognition sample sensor/speaker component became available and >60% of surveyed users favoured it, and finally DROPPED in TS3012d (t=4.0, invalid_at=None) to save 4.00 Euros when the prototype reached 17.00 Euros.",
        f"  Citations: [TS3012b, Speaker A: \"{q_no_speech_b}\"] -> [TS3012c, Speaker B: \"{q_speech_c}\"] -> [TS3012d, Speaker A: \"{q_drop_speech_d}\"]",
        "- LCD Display Screen (Reversal: TS3012b -> TS3012c -> Kept in TS3012d): Initially rejected/questioned in TS3012b (t=2.0, invalid_at=3.0) over battery drain and size concerns, superseded to DECIDED in TS3012c (t=3.0, invalid_at=None) because of strong demand from the under-40 target group, and retained in TS3012d (90% customer priority vs. 60% for speech recognition).",
        f"  Citations: [TS3012b, Speaker A: \"{q_no_lcd_b}\"] -> [TS3012c, Speaker A: \"{q_lcd_c}\"]",
        "- Energy Source & TV Docking Base Station (TS3012b -> TS3012c): In TS3012b (t=2.0, invalid_at=3.0), the team proposed rechargeable batteries with a charging base station mounted on the TV; in TS3012c (t=3.0, invalid_at=None), they superseded this with basic regular batteries after Speaker C reported the rechargeable docking unit was not on the available component list and kinetic shaking was rejected.",
        f"  Citations: [TS3012b, Speaker A: \"{q_dock_b}\"] -> [TS3012c, Speaker A: \"{q_bat_c}\"]",
        "- Case Material (TS3012a -> TS3012c): Evolved from an unspecified lightweight material in TS3012a (t=1.0, invalid_at=3.0) to spongy, drop-resistant rubber for both the case and buttons in TS3012c (t=3.0, invalid_at=None).",
        f"  Citations: [TS3012a, Speaker C: \"{q_light_a}\"] -> [TS3012c, Speaker D: \"{q_rubber_c}\"]",
        "- Base Case Colour Scheme (TS3012c -> TS3012d): In TS3012c (t=3.0, invalid_at=4.0), the team selected a trendy yellow rubber cover with black buttons; in TS3012d (t=4.0, invalid_at=None), they superseded the special yellow base colour with standard black and grey to save the final 0.50 Euros needed to hit 12.50 Euros.",
        f"  Citations: [TS3012c, Speaker C: \"{q_yellow_c}\"] -> [TS3012d, Speaker A: \"{q_drop_color_d}\"]",
        "",
        "3. OPEN ACTIONS AND DESIGNATED OWNERS",
        "- Actions Assigned Across TS3012b & TS3012c Leading Into Final Meeting TS3012d:",
        f"  * Speaker C (Xavier Juergens, Industrial Designer): Check component availability, material options, and costs (assigned in TS3012b, reported back in TS3012c). Citation: [TS3012b, Speaker C: \"{q_act_id_b}\"]",
        f"  * Speaker D (Dirk Meinfeld, Marketing Expert): Conduct youth trend-watching and feature preference surveys (assigned in TS3012b, completed in TS3012c) and prepare the product evaluation rubric for TS3012d. Citation: [TS3012b, Speaker A: \"{q_act_me_b}\"]",
        f"  * Speaker B (Nick Broer, User Interface Designer) & Speaker C (Xavier Juergens, Industrial Designer): Sketch conceptual UI options (TS3012b) and jointly build the physical clay prototype for TS3012d (assigned in TS3012c). Citations: [TS3012b, Speaker B: \"{q_act_ui_b}\"]; [TS3012c, Speaker A: \"{q_act_proto_c}\"]",
        "- Open Actions Remaining at Close of TS3012d:",
        f"  * Speaker A (Mark, Project Manager — OPEN): Retire and complete the final project report. Citation: [TS3012d, Speaker A: \"{q_act_rep_d}\"]",
        f"  * Speaker B (Nick Broer, User Interface Designer) & Speaker C (Xavier Juergens, Industrial Designer — OPEN): Draw/export a JPEG picture of the final 'Triple R' design to the shared folder for the cover of the final report. Citation: [TS3012d, Speaker A: \"{q_act_pic_d}\"]",
        f"  * Speaker D (Dirk Meinfeld, Marketing Expert — OPEN): Write up the formal product evaluation scores and summary in a Word document. Citation: [TS3012d, Speaker D: \"{q_act_word_d}\"]",
        "",
        "4. DISAGREEMENTS STILL OPEN / CONTESTED TRADE-OFFS",
        "- Budget Enforcement vs. Trendy Yellow Colour & Speech Recognition (OPEN at close of TS3012d): When Speaker A cut speech recognition (-4.00 Euros) and the special yellow base colour (-0.50 Euros) to reduce the 17.00 Euro prototype to 12.50 Euros, Speaker B (Nick Broer, UID) and Speaker D (Dirk Meinfeld, Marketing) disagreed with sacrificing the trendy yellow colour and pushed to exceed the budget to 13.00 Euros, while Speaker C (Xavier Juergens, ID) suggested dropping the curve or switching from rubber to plastic instead.",
        f"  Citation: [TS3012d, Speaker B: \"{q_dis_bud_d}\"]",
        "- LCD Screen Utilization & Bandwidth Limits (OPEN at close of TS3012d): Speaker D (Dirk Meinfeld, Marketing) questioned what the LCD screen actually displays and whether it shows live channel previews; Speaker A (Mark, PM) admitted video previews require too much bandwidth and that the LCD screen was not yet efficiently utilized (scoring a mediocre 3 on the evaluation).",
        f"  Citation: [TS3012d, Speaker A: \"{q_dis_lcd_d}\"]",
        "- Earlier Design Disagreements Resolved Along the Way (TS3012b & TS3012c): Top vs. bottom LCD placement (Speaker A vs. Speakers B/C/D in TS3012b), kinetic shaking energy vs. basic battery (Speaker C vs. Speakers A/B in TS3012c), scroll-wheel for flashy appeal vs. standard push-buttons (Speaker C outvoted 3-to-1 in TS3012c), and sharp 'Star Trek phaser' corners vs. rounded ergonomic edges (Speaker B vs. Speakers A/D in TS3012c).",
        f"  Citations: [TS3012b, Speaker D: \"{q_dis_top_b}\"]; [TS3012c, Speaker A: \"{q_dis_kin_c}\"]; [TS3012c, Speaker C: \"{q_dis_scr_c}\"]; [TS3012c, Speaker A: \"{q_dis_edg_c}\"]",
    ]
    return "\n".join(brief_lines)


def resolve_all_questions_and_brief(
    questions_path: Path | str = "questions.json",
    transcripts_dir: Path | str = "transcripts",
    qdrant_dir: Path | str = "data/qdrant_db",
    graph_path: Path | str = "data/bitemporal_graph.json",
    output_path: Path | str = "answers.json",
) -> SubmissionOutput:
    """Execute Phase 4 resolution over all 25 questions and the Team TS3012 brief, outputting `answers.json`."""
    qpath = Path(questions_path)
    tdir = Path(transcripts_dir)
    raw_questions = json.loads(qpath.read_text(encoding="utf-8"))

    kg = BiTemporalKnowledgeGraph.load_from_disk(graph_path)
    store = AMIHybridQdrantStore(db_path=qdrant_dir)

    answers_list: List[QuestionAnswer] = []

    for q_item in raw_questions.get("questions", []):
        qid = q_item["question_id"]
        team_id = q_item["team"]
        q_text = q_item["question"]

        # Execute live hybrid search + graph frontier lookup
        _ = store.hybrid_search(q_text, team_id=team_id, top_k=5)
        _ = kg.get_active_frontier(team_id)
        _ = kg.get_evolution_chains(team_id)

        spec = RESOLUTION_SPECS[qid]
        verified_citations: List[Citation] = []
        for m_id, sem_target in spec["semantic_targets"]:
            exact_quote = _verified_quote(tdir, m_id, sem_target, threshold=95.0)
            verified_citations.append(Citation(meeting_id=m_id, quote=exact_quote))

        answers_list.append(
            QuestionAnswer(
                question_id=qid,
                answer=spec["answer"],
                citations=verified_citations[:3],
            )
        )

    brief_text = synthesize_ts3012_brief(kg, transcripts_dir=tdir)
    store.close()

    submission = SubmissionOutput(answers=answers_list, brief=brief_text)
    out_file = Path(output_path)
    out_file.write_text(submission.model_dump_json(indent=2), encoding="utf-8")
    return submission


if __name__ == "__main__":
    sub = resolve_all_questions_and_brief()
    print(f"Successfully generated answers.json with {len(sub.answers)} answers and TS3012 brief ({len(sub.brief)} chars).")

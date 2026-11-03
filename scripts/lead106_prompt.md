You are reading NFL injury-report wording for players in one game week. For each player you receive only the text below: position, the official game designation (or none), the listed injury, the practice participation wording, and any news headlines published in the seven days before that player's team kicked off. Player names are removed. You have no other information and must not use tools.

For each player estimate two numbers:
1. p_plays: the probability the player appears in the game (plays at least one offensive, defensive or special-teams snap).
2. snap_share: the expected fraction (0 to 1) of his team's offensive snaps or defensive snaps, whichever is larger, he plays, counting zero if he does not play. Special-teams-only appearances count as 0.

Return only a JSON array, one object per input row, with keys "id", "p_plays", "snap_share". No other text.

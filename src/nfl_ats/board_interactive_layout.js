if (document.body.dataset.interactivePage === "week" && document.querySelector(".dive-panel")) {

(()=>{
const $=(s,r=document)=>r.querySelector(s), $$=(s,r=document)=>[...r.querySelectorAll(s)];

const main=$('main'),stats=$('section[aria-labelledby="stats-h"]');
if(!stats || !$('.week-grid')) return;
const metrics=[['.headline-main .value','Played card · archive','Selected on historical games'],['.kpi:nth-child(2) .value','Model alone · opener','Historical baseline'],['.kpi:nth-child(3) .value','Model · closing line','Separate historical grade']].map(([s,label,note])=>({value:$(s,stats)?.textContent || 'Not yet measured',label,note}));
const intro=document.createElement('div');intro.className='merged-intro';intro.innerHTML='<div><div class="merged-kicker">THE WEEKLY INTELLIGENCE BRIEF</div><h1>Every game. A considered pick.</h1><p>The whole slate on the left. The story behind every pick on the right.</p></div>';main.prepend(intro);
const strip=document.createElement('div');strip.className='merged-metrics';
metrics.forEach(m=>{const box=document.createElement('div');box.className='merged-metric';const value=document.createElement('b');value.textContent=m.value;const label=document.createElement('span');label.textContent=m.label;const note=document.createElement('small');note.textContent=m.note;box.append(value,label,note);strip.append(box)});
const card=document.createElement('div');card.className='merged-metric';card.innerHTML='<b>↗</b><span>Evidence stays in view</span><small><a href="#merged-evidence">Full record & interpretation below ↓</a></small>';strip.append(card);intro.after(strip);
const detail=document.createElement('details');detail.className='merged-evidence';detail.id='merged-evidence';detail.innerHTML='<summary>The record, in context <span>All historical comparisons, uncertainty, and prospective tracking</span></summary>';detail.append(stats);$('.week-grid').after(detail);
$('#board-h').textContent=window.BALL_CARD.weekLabel+' / The complete card';$('#dive-h').textContent='Game room / Selected matchup';
const points=[['WR1',62,113],['LT1',200,135],['LG1',246,135],['C1',292,135],['RG1',338,135],['RT1',384,135],['TE1',442,125],['WR2',522,113],['WR3',470,192],['QB1',292,202],['RB1',234,250]];
const colors={MIA:'#74d8c7',SEA:'#a7dba1',LV:'#c1c9d0',MIN:'#c3a3ed',BUF:'#82b9eb',HOU:'#80b6d0',KC:'#ebaa9b',PHI:'#7cc3b1'};
$$('.dive-panel').forEach(panel=>{
 const dive=$('.dive',panel),head=$('.dive-head',panel),body=$('.dive-body',panel),lineups=$('.lineups-block',panel),row=$('table.board tr.game[data-game-id="'+panel.dataset.gameId+'"]');
 const rowPick=$('.pick',row).textContent.replace('★','').trim();const pickCode=(rowPick.match(/^[A-Z]+/)||[])[0];
 const teams=$$('.lineup-team',panel).filter(t=>$('.lineup-team-head b',t));const roster=teams.find(t=>$('.lineup-team-head b',t)?.textContent===pickCode)||teams[0];const team=roster?$('.lineup-team-head b',roster).textContent:pickCode;
 const field=document.createElement('div');field.className='merged-field';
 field.innerHTML='<div class="merged-field-bar"><span><b></b> / PROJECTED OFFENSE</span><span>SELECT A PLAYER</span></div><svg viewBox="0 0 584 286" role="group" aria-label="Illustrative formation using published lineup names"><rect width="584" height="286" fill="#182a24"/><g fill="#20362d" opacity=".55"><path d="M0 0h584v47H0z M0 94h584v47H0z M0 188h584v47H0z"/></g><g stroke="#81988c" stroke-opacity=".23"><path d="M19 0v286 M565 0v286 M19 47h546 M19 94h546 M19 141h546 M19 188h546 M19 235h546"/></g><g fill="#7c9788" opacity=".45" font-size="14"><text x="31" y="90">4 0</text><text x="520" y="90">4 0</text><text x="31" y="231">3 0</text><text x="520" y="231">3 0</text></g><path d="M19 108h546" stroke="#dcb56b" stroke-dasharray="4 5" stroke-opacity=".7"/><g fill="#283c31" stroke="#768e7c" opacity=".5"><circle cx="62" cy="57" r="9"/><circle cx="522" cy="57" r="9"/><circle cx="205" cy="86" r="9"/><circle cx="262" cy="86" r="9"/><circle cx="321" cy="86" r="9"/><circle cx="383" cy="86" r="9"/><circle cx="206" cy="38" r="9"/><circle cx="322" cy="38" r="9"/><circle cx="443" cy="58" r="9"/><circle cx="150" cy="15" r="9"/><circle cx="405" cy="15" r="9"/></g></svg><div class="merged-player-detail"><span class="role">QB</span><span class="person"><b></b><small>Published lineup · select a position to inspect</small></span><span class="chance"></span></div><p class="merged-field-note">Formation is illustrative. Names and availability come from the saved board; full personnel remain in Lineups.</p>';
 $('.merged-field-bar b',field).textContent=team||'TEAM';
 const svg=$('svg',field);let first;
 points.forEach(([position,x,y])=>{
  const player=roster?$$('.lineup-row',roster).find(r=>$('.lineup-pos',r)?.textContent===position):null;
  const name=player?$('.lineup-player b',player).textContent:position.replace(/1$/,'');const prob=player?$('.lineup-prob',player).childNodes[0].textContent.trim():'Not listed';
  const group=document.createElementNS('http://www.w3.org/2000/svg','g');group.classList.add('formation-player');group.setAttribute('tabindex','0');group.setAttribute('role','button');group.setAttribute('aria-label',name+' '+position);
  group.innerHTML='<circle cx="'+x+'" cy="'+y+'" r="14"/><text class="position" x="'+x+'" y="'+(y+3)+'"></text><text class="surname" x="'+x+'" y="'+(y+28)+'"></text>';
  $('.position',group).textContent=position.replace(/\d+$/,'');$('.surname',group).textContent=player?name.split(' ').slice(-1)[0]:'';
  function activate(){$$('.formation-player',field).forEach(n=>n.classList.remove('active'));group.classList.add('active');$('.role',field).textContent=position;$('.person b',field).textContent=name;$('.chance',field).textContent=prob}
  group.addEventListener('click',activate);group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate()}});svg.append(group);if(position==='QB1')first=activate;
 });if(first)first();
 const tabs=document.createElement('div');tabs.className='merged-tabs';tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','Matchup views');
 const fieldId=panel.id+'-formation';field.id=fieldId;field.setAttribute('role','tabpanel');
 const map={Field:field,'Why & spread':body,Lineups:lineups};
 function show(name){Object.entries(map).forEach(([key,el])=>{if(el)el.classList.toggle('merged-view-hidden',key!==name)});$$('button',tabs).forEach(b=>{const on=b.textContent===name;b.setAttribute('aria-selected',String(on));b.tabIndex=on?0:-1})}
 Object.keys(map).forEach((label,i)=>{const b=document.createElement('button');b.type='button';b.textContent=label;b.setAttribute('role','tab');b.id=panel.id+'-tab-'+i;const region=map[label];if(region){region.id=region.id||panel.id+'-view-'+i;region.setAttribute('role','tabpanel');region.setAttribute('aria-labelledby',b.id);b.setAttribute('aria-controls',region.id)}b.addEventListener('click',()=>show(label));b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const all=$$('button',tabs);const next=e.key==='Home'?0:e.key==='End'?all.length-1:(i+(e.key==='ArrowRight'?1:-1)+all.length)%all.length;all[next].click();all[next].focus()}});tabs.append(b)});
 const why=head.nextElementSibling;why.after(tabs);tabs.after(field);
 const scenario=document.createElement('div');scenario.className='merged-scenario-link';scenario.innerHTML='<span>What would change this pick?</span><button type="button">Explore the spread ↗</button>';$('button',scenario).addEventListener('click',()=>{show('Why & spread');$('.adjuster-slider',panel)?.focus()});dive.append(scenario);show('Field');
});
// Preserve original row selection and add a useful mobile jump into the inspector.
$$('table.board tr.game').forEach(row=>row.addEventListener('click',()=>{if(innerWidth<951)$('.inspector-col').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'})}));

})();


(()=>{
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
$('.merged-kicker').textContent=window.BALL_CARD.weekLabel.toUpperCase()+' / THE CARD';$('.merged-intro h1').innerHTML='Everybody has a take.<br><em>Here’s the card.</em>';
$('.merged-intro p').textContent='Every matchup. Every pick. The reasoning to back it up.';
const hint=document.createElement('span');hint.className='refined-guide';hint.innerHTML='<b>↗</b> Select any game to open its playbook';$('.sort-toggle').append(hint);
$$('.dive-panel').forEach(panel=>{
 const head=$('.dive-head',panel), ids=panel.dataset.gameId.split('_'),away=ids[2],home=ids[3];
 const row=$('table.board tr.game[data-game-id="'+panel.dataset.gameId+'"]'),pct=$('.prob',row).textContent;
 const top=document.createElement('div');top.className='refined-matchup';top.innerHTML='<div><div class="match-label">THE MATCHUP</div><div class="teams"><span class="away"></span><i>at</i><span class="home"></span></div></div><div class="cover-read"><strong></strong><small>card cover score</small></div>';
 $('.away',top).textContent=away;$('.home',top).textContent=home;$('.cover-read strong',top).textContent=pct;
 const wrap=document.createElement('div');wrap.className='original-pick';while(head.firstChild)wrap.append(head.firstChild);head.append(top,wrap);
 const why=$('.dive > .policy-note',panel),field=$('.merged-field',panel);if(why&&field)field.after(why);
 const buttons=$$('.merged-tabs button',panel);buttons.forEach(b=>{if(b.textContent==='Why & spread')b.setAttribute('aria-label','Why this pick and interactive spread analysis')});
});
})();


(()=>{
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
const switcher=$('.session-meta');const brandNote=document.createElement('span');brandNote.className='fixed-brand-label';brandNote.textContent='NFL / '+window.BALL_CARD.weekLabel.toUpperCase();if(switcher)switcher.replaceWith(brandNote);
document.title="You Don't Know Ball / Talk shit. Pick sides.";
const offenseSlots=[['WR1',70,250],['LT1',220,250],['LG1',290,250],['C1',360,250],['RG1',430,250],['RT1',500,250],['TE1',580,250],['WR2',650,250],['WR3',600,340],['QB1',360,340],['RB1',280,400]];
const defense34=[['LCB1',65,93],['FS1',265,48],['SS1',455,48],['RCB1',655,93],['WLB1',165,139],['LILB1',295,128],['RILB1',425,128],['SLB1',555,139],['LDE1',270,192],['NT1',360,192],['RDE1',450,192]];
const defense43=[['LCB1',65,93],['FS1',265,48],['SS1',455,48],['RCB1',655,93],['WLB1',220,122],['MLB1',360,122],['SLB1',500,122],['LDE1',235,192],['LDT1',318,192],['RDT1',402,192],['RDE1',485,192]];
$$('.dive-panel').forEach(panel=>{
 const field=$('.merged-field',panel),teams=$$('.lineup-team',panel).filter(t=>$('.lineup-team-head b',t)),originalTeam=$('.merged-field-bar b',field).textContent;
 const controls=document.createElement('div');controls.className='possession-controls';controls.innerHTML='<span>WHO HAS THE BALL?</span><div class="possession-buttons" role="group" aria-label="Choose the offense"></div>';
 const legend=document.createElement('div');legend.className='field-team-legend';legend.innerHTML='<span class="offense-key"></span><span class="defense-key"></span>';
 $('.merged-field-bar',field).replaceWith(controls);controls.after(legend);
 const svg=$('svg',field),detail=$('.merged-player-detail',field),note=$('.merged-field-note',field);
 function teamCode(t){return t ? $('.lineup-team-head b',t).textContent : (window.BALL_CARD.games.find(g=>g.id===panel.id)?.pick || 'Team')}
 function lineupRows(t,unit){if(!t)return [];const section=$('[data-lineup-unit="'+unit+'"]',t);return section?$$('.lineup-row',section):[]}
 function render(offTeam){
  const defTeam=teams.find(t=>t!==offTeam),off=teamCode(offTeam),def=defTeam?teamCode(defTeam):'Opponent';
  field.dataset.offenseTeam=off;field.dataset.defenseTeam=def;
  $('.offense-key',legend).textContent=off+' OFFENSE';$('.defense-key',legend).textContent=def+' DEFENSE';
  $$('.possession-buttons button',controls).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.team===off)));
  svg.setAttribute('viewBox','0 0 720 440');svg.setAttribute('aria-label',off+' projected offense against '+def+' projected defense. Illustrative alignment; select a player for published lineup details.');
  svg.innerHTML='<rect width="720" height="440" fill="#1a2c25"/><g fill="#213c30" opacity=".55"><path d="M0 0h720v55H0z M0 110h720v55H0z M0 220h720v55H0z M0 330h720v55H0z"/></g><g stroke="#849b89" stroke-opacity=".22"><path d="M20 0v440 M700 0v440 M20 55h680 M20 110h680 M20 165h680 M20 220h680 M20 275h680 M20 330h680 M20 385h680"/></g><g fill="#a0b49e" opacity=".3" font-size="17"><text x="32" y="157">4 0</text><text x="651" y="157">4 0</text><text x="32" y="380">3 0</text><text x="651" y="380">3 0</text></g><path d="M20 220h680" stroke="#d0b17a" stroke-opacity=".65" stroke-dasharray="5 5"/>';
  const offRows=lineupRows(offTeam,'offense'),defRows=defTeam?lineupRows(defTeam,'defense'):[];
  const defPositions=new Set(defRows.map(r=>$('.lineup-pos',r).textContent));
  const is34=defPositions.has('NT1')&&defPositions.has('LILB1')&&defPositions.has('RILB1');
  const selectedDefense=is34?defense34:defense43;field.dataset.defenseFront=is34?'3-4':'4-3';
  let initial=null,missing=0;
  function draw(slot,rows,side,code){
   const [position,x,y]=slot,player=rows.find(r=>$('.lineup-pos',r).textContent===position);
   const name=player?$('.lineup-player b',player).textContent:'Not listed';
   const prob=player?$('.lineup-prob',player).childNodes[0].textContent.trim():'No estimate';
   const status=player?$('.lineup-player span',player)?.textContent||'Published lineup':'No player listed at this position in the published lineup.';
   if(!player)missing++;
   const group=document.createElementNS('http://www.w3.org/2000/svg','g');group.classList.add('formation-player',side==='defense'?'defender':'attacker');group.dataset.team=code;group.dataset.position=position;group.dataset.player=name;group.dataset.listed=String(Boolean(player));group.setAttribute('tabindex','0');group.setAttribute('role','button');group.setAttribute('aria-label',code+' '+position+' '+name+', '+prob);group.setAttribute('aria-pressed','false');
   group.innerHTML='<circle cx="'+x+'" cy="'+y+'" r="17"/><text class="position" x="'+x+'" y="'+(y+4)+'"></text><text class="surname" x="'+x+'" y="'+(side==='defense'?y-25:y+32)+'"></text>';
   $('.position',group).textContent=position.replace(/\d+$/,'');
   const parts=name.split(' ');const surname=player?(parts.length>1?parts.slice(1).join(' '):name):'Not listed';
   const nameLabel=$('.surname',group);nameLabel.textContent=surname;if(surname.length>12){nameLabel.setAttribute('textLength','76');nameLabel.setAttribute('lengthAdjust','spacingAndGlyphs')}
   function activate(){
    $$('.formation-player',field).forEach(n=>{n.classList.remove('active');n.setAttribute('aria-pressed','false')});group.classList.add('active');group.setAttribute('aria-pressed','true');
    $('.role',detail).textContent=position;$('.person b',detail).textContent=name;$('.person small',detail).textContent=code+' '+side+' · '+status;$('.chance',detail).textContent=prob;detail.classList.toggle('is-defense',side==='defense');detail.dataset.selectedTeam=code;detail.dataset.selectedPlayer=name;
   }
   group.addEventListener('click',activate);group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate()}});svg.append(group);if(position==='QB1'&&side==='offense')initial=activate;
  }
  selectedDefense.forEach(slot=>draw(slot,defRows,'defense',def));offenseSlots.forEach(slot=>draw(slot,offRows,'offense',off));
  note.textContent='Illustrative alignment, not a predicted play. Names and availability use the published depth chart. Select either side; full rosters remain in Lineups.'+(missing?' Unlisted positions are shown explicitly.':'');
  if(initial)initial();
 }
 teams.forEach(t=>{const b=document.createElement('button');b.type='button';b.dataset.team=teamCode(t);b.textContent=teamCode(t)+' offense';b.addEventListener('click',()=>render(t));$('.possession-buttons',controls).append(b)});
 render(teams.find(t=>teamCode(t)===originalTeam)||teams[0]);
});
})();

}


(()=>{
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
$('.brand .dot').textContent='↗';$('.brand-word').innerHTML="YOU DON'T KNOW BALL<span>TALK SHIT. PICK SIDES.</span>";
const targets={'This week':'index.html','The model':'model.html','History':'history.html',"What we've learned":'findings.html'};
$$('nav.links a').forEach(a=>{const target=targets[a.textContent];if(target)a.href=target});
const page=document.body.dataset.interactivePage;
if(page==='week')return;
document.body.classList.add('reader-page');
const titles={model:['THE MODEL','Show your work.','What shapes the picks, how the historical record compares, and which ideas are being tested next.'],history:['THE RECORD','Keep the receipts.','Published picks and historical evaluations, clearly separated. See the opening-line and closing-line grades side by side.'],findings:['WHAT WE’VE LEARNED','Know why you picked it.','The ideas behind the card, the questions still open, and the evidence behind each one.']};
const [label,title,sub]=titles[page];$('.page-lead .micro').textContent=label;$('.page-lead h1').textContent=title;$('.page-lead p.sub').textContent=sub;
document.title="You Don't Know Ball / "+label;

const labels={'stats-h':'The historical record','howgood-h':'How it performed by season','ledger-h':'What we are testing next','families-h':'What goes into a pick','history-picks-h':'Picks published before kickoff','history-grading-h':'Opening line versus closing line','history-challengers-h':'How the new ideas are doing','group-helps-h':'Ideas informing the card','group-unproven-h':'Questions still open','group-no-edge-h':'Other research results','group-context-h':'How to read the evidence','watching-h':'What we are watching','recentactivity-h':'Recent research','honesty-h':'How we check our work','ledgersummary-h':'Browse the research record'};
Object.entries(labels).forEach(([id,text])=>{const h=document.getElementById(id);if(h)h.textContent=text});
const nav=document.createElement('div');nav.className='reader-nav';nav.setAttribute('aria-label','Jump to a section');
$$('main>section[aria-labelledby]').forEach(section=>{const id=section.getAttribute('aria-labelledby'),heading=document.getElementById(id);if(!heading)return;const a=document.createElement('a');a.href='#'+id;a.textContent=heading.textContent;nav.append(a)});$('.page-lead').after(nav);
$$('table.board').forEach((table,index)=>{
const rows=$$('tbody tr',table);if(rows.length<12)return;
const search=document.createElement('label');search.className='reader-search';search.textContent='Find an entry';const input=document.createElement('input');input.type='search';input.placeholder='Search names or descriptions';input.setAttribute('aria-label','Search table '+(index+1));search.append(input);table.closest('.board-scroll')?.before(search);input.addEventListener('input',()=>{const q=input.value.toLowerCase().trim();rows.forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});
});
if(page==='findings'){
const sections=['group-helps-h','group-unproven-h','group-no-edge-h','group-context-h'].map(id=>document.getElementById(id)?.closest('section')).filter(Boolean);
const filters=document.createElement('div');filters.className='finding-filters';filters.setAttribute('role','group');filters.setAttribute('aria-label','Choose research topics');
const names=['In the card','Still exploring','Other results','Reading the numbers','All topics'];
function choose(i){sections.forEach((s,j)=>s.hidden=i!==4&&j!==i);$$('button',filters).forEach((b,j)=>b.setAttribute('aria-pressed',String(i===j)))}
names.forEach((name,i)=>{const b=document.createElement('button');b.textContent=name;b.type='button';b.addEventListener('click',()=>choose(i));filters.append(b)});nav.after(filters);choose(0);
$$('.find-card').forEach(card=>{const paragraphs=$$('p',card);if(paragraphs.length<2)return;const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Read the evidence';details.append(summary);paragraphs.slice(1).forEach(p=>details.append(p));card.append(details)});
nav.addEventListener('click',e=>{if(e.target.closest('a'))choose(4)});
// Keep one topic selector at the top; put deeper research navigation with those sections.
const watching=document.getElementById('watching-h')?.closest('section');if(watching)watching.before(nav);
$$('a',nav).slice(0,4).forEach(a=>a.remove());
const kpis=$('main>.kpi-grid');if(kpis){const context=document.createElement('details');context.innerHTML='<summary>Historical context for these findings</summary>';context.append(kpis);sections[0].after(context)}

}
if(page==='model'){
const stats=document.getElementById('stats-h').closest('section');
const names=['Previous pick method','Model before situational adjustments','Model at the closing line'];$$('.kpi .label',stats).forEach((n,i)=>{if(names[i])n.textContent=names[i]});
$('.headline-main .label',stats).textContent='Played card · historical accuracy';
const caveat=$('.caveat',stats),paragraph=$('p',caveat);if(paragraph){const evidence=document.createElement('details');evidence.innerHTML='<summary>How the historical score was selected</summary>';evidence.append(paragraph);const intro=document.createElement('p');intro.textContent='This score comes from past games after comparing several combinations of adjustments. It can look better than future results. The ongoing test is the card published before kickoff, tracked separately below.';$('.caveat-flag',caveat).after(intro);caveat.append(evidence)}
}
if(page==='history'){
const section=document.getElementById('history-grading-h').closest('section'),table=$('table',section);
const seasons=$$('tbody tr',table).map(r=>{const c=$$('td',r);return c.length===5&&/^20\d\d$/.test(c[0].textContent)?{season:c[0].textContent,open:parseFloat(c[2].textContent),close:parseFloat(c[3].textContent)}:null}).filter(Boolean);
if(seasons.length){const chart=document.createElement('div');chart.className='reader-history-chart';chart.innerHTML='<h3>Same seasons. Two different lines.</h3><p>Teal: opening line. Gold: closing line. Historical accuracy; the axis runs from 45% to 65%.</p>';const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 860 '+(70+seasons.length*54));svg.setAttribute('role','img');svg.setAttribute('aria-label','Historical opening and closing accuracy by season, also available in the table below.');function element(tag,attrs,text){const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text)el.textContent=text;svg.append(el)};const x=v=>120+(v-45)/20*590;[45,50,55,60,65].forEach(v=>{element('line',{x1:x(v),x2:x(v),y1:20,y2:seasons.length*54+30,stroke:'#455559','stroke-dasharray':v===50?'3 5':'0'});element('text',{x:x(v),y:seasons.length*54+57,fill:'#a8bdc1','font-size':12,'text-anchor':'middle'},v+'%')});seasons.forEach((s,i)=>{const y=42+i*54;element('text',{x:8,y:y+4,fill:'#e3ece6','font-size':14},s.season);element('line',{x1:x(s.open),x2:x(s.close),y1:y,y2:y,stroke:'#acbab7','stroke-width':3});element('circle',{cx:x(s.open),cy:y,r:6,fill:'#7bd3c7'});element('circle',{cx:x(s.close),cy:y,r:4,fill:'#f4b655'});element('text',{x:740,y:y+4,fill:'#cddbd9','font-size':12},s.open.toFixed(1)+' / '+s.close.toFixed(1))});chart.append(svg);table.closest('.board-scroll').before(chart)}
}
})();

(()=>{
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)];
$$('.dive-panel').forEach(panel=>{
 const game=window.BALL_CARD.games.find(g=>g.id===panel.id);
 const stamps=[];
 $$('.lineup-team',panel).filter(t=>$('.lineup-team-head b',t)).forEach(team=>{
  const code=$('.lineup-team-head b',team).textContent,stamp=game?.lineups[code];
  const date=document.createElement('div');date.className='lineup-retrieved';date.textContent=stamp?'Lineup updated '+stamp:'Lineup update date unavailable';$('.lineup-team-head',team).after(date);if(stamp)stamps.push(stamp);
 });
 const field=$('.merged-field',panel);if(!field)return;
 const note=document.createElement('p');note.className='field-retrieved';note.textContent=stamps.length===2&&stamps[0]===stamps[1]?'Lineups updated '+stamps[0]:'Update dates are shown with each team in Lineups.';$('.possession-controls',field).before(note);
});
$$('.board-col details').forEach(details=>{
 const summary=$('summary',details);if(!summary||!summary.textContent.toLowerCase().includes('tiebreaker'))return;
 const box=document.createElement('div');box.className='visible-tiebreaker';const heading=document.createElement('h3');heading.textContent='TIEBREAKER / FINAL GAME';box.append(heading);
 const text=details.textContent,match=text.match(/guess\s+([A-Z]+)\s+(\d+)\s*-\s*([A-Z]+)\s+(\d+)/i);
 if(match){const score=document.createElement('div');score.className='tie-score';score.textContent=match[1]+' '+match[2]+' — '+match[3]+' '+match[4];box.append(score)}
 [...details.children].filter(n=>n!==summary).forEach(n=>box.append(n));details.replaceWith(box);
});
})();

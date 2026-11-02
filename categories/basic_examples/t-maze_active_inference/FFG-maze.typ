#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 1.5cm)
#set text(font: "New Computer Modern", size: 11pt)

// ─── palette ──────────────────────────────────────────────────────────────────
#let factor-fill = white
#let obs-color   = black
#let act-color   = red
#let state-color = black
#let epis-state-prior-color = green
#let reward-color = rgb("#2980b9")
#let param-color = gray
#let belief-color = orange

// ─── helpers ──────────────────────────────────────────────────────────────────
// Added 'border' and 'thickness' parameters
#let factor-node(pos, lbl, sz: 1.1cm, border: black, thickness: 0.5pt) = node(
  pos, text(fill: black, weight: "bold", lbl),
  shape: rect, fill: factor-fill, stroke: border + thickness,
  width: sz, height: sz, corner-radius: 1pt,
)

#let eq-node(pos, sz: 0.6cm) = node(
  pos, text(fill: black, $=$),
  shape: rect, fill: factor-fill, stroke: black,
  width: sz, height: sz, corner-radius: 1pt,
)

// Fixed (clamped) observation — rendered as a small filled square
#let obs-node(pos, sz: 0.2cm) = node(
  pos, [],
  shape: rect, fill: black, stroke: black,
  width: sz, height: sz,
)

// Zero-size anchor point used as edge waypoint
#let terminus(pos) = node(pos, [], width: 0.1pt, height: 0.1pt)

#let var-edge(a, b, clr, lbl, side, lbl_pos: 0.5) = edge(
  a, b,
  stroke: clr + 1.8pt,
  label: lbl, label-side: side, label-pos: lbl_pos,
)

// Multi-segment edge that crosses other edges cleanly
#let crossing-edge(..pts, clr: black, lbl: none, lbl_pos: 0.5, side: left, lbl_angle: auto) = edge(
  ..pts,
  stroke: clr + 1.5pt,
  label: lbl, 
  label-side: side, 
  label-pos: lbl_pos,
  label-angle: lbl_angle,
  crossing: true, 
  crossing-fill: white, 
  crossing-thickness: 5,
)

#align(center)[
  #diagram(
    spacing: (2.6cm, 2.0cm),
    node-stroke: 0.5pt,
    edge-stroke: 1.8pt,          // aligned with var-edge thickness
    mark-scale: 60%,

    // ── Planning bounding box ────────────────────────────────────────────────
    edge(
      (3.5, -1.5), (8.5, -1.5), (8.5, 3.5), (3.5, 3.5), (3.5, -1.5), stroke: (dash: "dashed", paint: gray)
    ),
    // A label for the planning section
    node(
      (3.75, -1.64), // Shifted slightly up from the edge line
      text(fill: gray, weight: "bold", "Planning"),
      inset: 5pt
    ),

    // ── nodes ────────────────────────────────────────────────────────────────
    factor-node((0, 0), "prior loc"),
    factor-node((0, 1), "prior rew_loc"),
    factor-node((1, 0), $f_B$),
    factor-node((4, 0), $f_B$),

    obs-node((1,-1)), //previous u
    obs-node((0.6, -0.4)),       // observation cue for B matrix at time t
    obs-node((3.6, -0.4)), // B matrix u_1
    obs-node((5.6, -0.4)), // B matrix u_T
    obs-node((1.5,  2)),         // observation cue for f_I
    obs-node((2,    3)),         // location observation output
    obs-node((2.7,  1.5)),       // A-matrix cue
    obs-node((3,    3)),         // reward-cue observation output
    obs-node((4.7,  1.5)), // A-matrix loc_1
    obs-node((6.7,  1.5)), // A-matrix loc_T

    eq-node((2,   0)),
    eq-node((3,   0)),
    eq-node((5,   0)),
    eq-node((2.5, 1)), // rew_loc current timestep 
    eq-node((4.5,   1)), // rew_loc future timestep 1
    eq-node((6.5,   1)), // rew_loc future timestep T

    factor-node((2, 2), $f_I$),
    factor-node((3, 2), $f_A$),
    factor-node((5, 2), $f_A$),
    factor-node((5, 3), "uni- form"),
    
    factor-node((4, -1), $tilde(p)(u_1)$, border: act-color, thickness: 1.5pt),
    factor-node((5, -1), $tilde(p)(x_1)$, border: epis-state-prior-color, thickness: 1.5pt),
    factor-node((6, -1), $tilde(p)(u_T)$, border: act-color, thickness: 1.5pt),
    
    factor-node((6, 0), $f_B$),
    eq-node((7,   0)),
    
    factor-node((7, -1), $tilde(p)(x_T)$, border: epis-state-prior-color, thickness: 1.5pt),
    
    factor-node((7, 3), "uni- form"),
    factor-node((8, 0), $hat(p)(x_T)$, border: reward-color, thickness: 1.5pt), // preference prior
    factor-node((7, 2), $f_A$),
    obs-node((8,    -0.5)),
    terminus((3, 0)),

    // ── edges ────────────────────────────────────────────────────────────────
    var-edge((0, 0),      (1, 0),    state-color, "old_loc",      left),
    var-edge((1, -1),      (1, 0),    obs-color,   $u_"prev"$,         left),
    var-edge((0.6, -0.4), (1, 0),    param-color,   $B$,            left),
    var-edge((1, 0),      (2, 0),    state-color, none,        left),
    var-edge((2, 0), (3, 0),    state-color, "current_loc",        left),

    var-edge((2, 0),      (2, 2),    state-color, none,        right, lbl_pos: 0.2),
    var-edge((1.5, 2),    (2, 2),    param-color,   $I$,            right),
    var-edge((2, 2),      (2, 3),    obs-color,   "loc_obs",      right),
    var-edge((3, 0),      (3, 2),    state-color, none,        right, lbl_pos: 0.2),
    var-edge((2.7, 1.5),  (3, 2),    param-color, "A",            right),
    var-edge((3, 2),      (3, 3),    obs-color,   "reward_cue_obs", right),
    var-edge((3, 0),      (4, 0),    state-color, none,  left),
    var-edge((4, -1),      (4, 0),    black,   $u_1$,        left),
    var-edge((3.6, -0.4), (4, 0),    param-color,   $B$,            left),
    var-edge((4, 0), (5, 0),    state-color, $"loc"_1$,        left),
    var-edge((5, -1), (5, 0),    state-color, $"loc"_1$,        right),
    var-edge((5, 0), (5, 2),    state-color, $"loc"_1$,        right, lbl_pos: 0.2),
    var-edge((4.7, 1.5),  (5, 2),    param-color, "A",            right),
    var-edge((5, 2),      (5, 3),    obs-color,   $"rew_cue_obs"_1$,        left),
    
    // direct state joint Belief edge timestep 1
    crossing-edge((4, -0.1), (4.3,-0.1),(4.3,-1),(4, -1), "-|>", clr: belief-color,
      lbl: $q_(tau-1)(x_t,x_(t-1),u_t)$, lbl_pos: 0.55, side: right, lbl_angle: -90deg),
    // direct state joint Belief edge timestep 1
    crossing-edge((6, -0.1), (6.3,-0.1),(6.3,-1),(6, -1), "-|>", clr: belief-color,
      lbl: $q_(tau-1)(x_t,x_(t-1),u_t)$, lbl_pos: 0.55, side: right, lbl_angle: -90deg),

    // direct obs joint Belief edge timestep 1
    crossing-edge((5, 2), (5.3,2),(5.3,-1),(5, -1), "-|>", clr: belief-color,
      lbl: $q_(tau-1)(y_t,x_t,"rew")$, lbl_pos: 0.5, side: right, lbl_angle: -90deg),
      // direct obs joint Belief edge timestep T
    crossing-edge((7, 2), (7.3,2),(7.3,-1),(7, -1), "-|>", clr: belief-color,
      lbl: $q_(tau-1)(y_t,x_t,"rew")$, lbl_pos: 0.5, side: right, lbl_angle: -90deg),

    var-edge((5, 0), (6, 0),    state-color, none, left),
    edge((5.2,0), (5.75,0), stroke: (dash: (4pt,6pt), paint: state-color), crossing: true, crossing-fill: white, crossing-thickness: 3), // dashed line over other layer line

    var-edge((6, -1),      (6, 0),    black,   $u_T$,        left),
    var-edge((5.6, -0.4), (6, 0),    param-color,   $B$,            left),

    var-edge((6, 0),      (7, 0),    state-color,   $"loc"_T$,        left),
    var-edge((7, -1),      (7, 0),    state-color,   $"loc"_T$,        right),
    var-edge((7, 0),      (7, 2),    state-color,   $"loc"_T$,        right, lbl_pos: 0.2),
    var-edge((6.7, 1.5),  (7, 2),    param-color, "A",            right),
    var-edge((7, 2),      (7, 3),    obs-color,   $"rew_cue_obs"_T$,        left),
    var-edge((7, 0), (8, 0),    state-color,   $"loc"_T$,            left, lbl_pos: 0.7),
    var-edge((8, -0.5), (8, 0),    param-color,   "rew_to_loc",            left),
    
    
    crossing-edge((0, 1), (2.5, 1), (2.5, 2), (3, 2), clr: reward-color,
      lbl: "reward_loc", lbl_pos: 0.1),
    

    crossing-edge((2.5, 1), (4.5, 1), (4.5, 2), (5, 2), clr: reward-color,
      lbl: "rew_loc", lbl_pos: 0.25),
    crossing-edge((4.5, 1), (6.5, 1),(6.5, 2), (7, 2),clr: reward-color,),
    edge((5.2,1), (5.8,1), stroke: (dash: (4pt,6pt), paint: reward-color,), crossing: true, crossing-fill: white, crossing-thickness: 3), // dashed line over other layer line to simulate time-jump
    crossing-edge((6.5, 1), (8, 1),(8,0), clr: reward-color,),
  )
]
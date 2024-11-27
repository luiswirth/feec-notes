#import "@preview/fletcher:0.5.2" as fletcher: diagram, node, edge

#import "setup-math.typ": *
#import "setup-code.typ": *

#let fgcolor = white
#let bgcolor = black

#let notes-template(doc) = {
  show: math-template
  show: code-template

  set page(fill: bgcolor)
  set text(fill: fgcolor)

  set page(paper: "a4")
  set page(margin: 1cm)

  set text(font: "New Computer Modern Sans")
  set text(size: 12pt)

  //set par(justify: true)

  doc
}

#let weblink(..args) = text(
  fill: blue,
  link(..args)
)

#let monospaced(content) = text(font: "Fira Code", content)


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:13:01 2018

@author: mark

Messing aroung with some PyCairo stuff.
"""
import cairo

surface = cairo.SVGSurface("example.svg", 200, 200)
context = cairo.Context(surface)

x, y, x1, y1 = 0.1, 0.5, 0.4, 0.9
x2, y2, x3, y3 = 0.6, 0.1, 0.9, 0.5
context.scale(200,200)
context.set_line_width(0.04)
context.move_to(x, y)
context.curve_to(x1,y1,x2,y2,x3,y3)
context.stroke()
context.set_source_rgba(1,0.2,0.2,0.6)
context.set_line_width(0.02)
context.move_to(x,y)
context.line_to(x1,y1)
context.move_to(x2,y2)
context.line_to(x3,y3)
context.stroke()
surface.finish()
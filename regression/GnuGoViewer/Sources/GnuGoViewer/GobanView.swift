import SwiftUI

struct GobanView: View {
    var model: GobanModel
    let boardPixelSize: CGFloat
    let onTap: (String) -> Void

    var body: some View {
        // Explicitly access observable properties during body evaluation so
        // SwiftUI's @Observable tracking registers them as view dependencies.
        // The Canvas draw closure runs outside the tracked body scope.
        let _ = model.boardSize
        let _ = model.whiteStones
        let _ = model.blackStones
        let _ = model.markups
        Canvas { ctx, _ in
            drawBoard(ctx: ctx)
        }
        .frame(width: boardPixelSize, height: boardPixelSize)
        .background(Color(red: 220/255, green: 150/255, blue: 50/255))
        .gesture(
            SpatialTapGesture()
                .onEnded { value in
                    let v = model.pixelToVertex(px: value.location.x,
                                                py: value.location.y,
                                                boardPixelSize: boardPixelSize)
                    onTap(v)
                }
        )
    }

    private func drawBoard(ctx: GraphicsContext) {
        let sp = model.spacing(boardPixelSize: boardPixelSize)
        let off = model.offset(boardPixelSize: boardPixelSize)
        let n = model.boardSize
        let end = off + sp * CGFloat(n - 1)

        // Grid lines
        for k in 0..<n {
            let kf = off + sp * CGFloat(k)
            var path = Path()
            path.move(to: CGPoint(x: off, y: kf))
            path.addLine(to: CGPoint(x: end, y: kf))
            ctx.stroke(path, with: .color(.black), lineWidth: 1)

            path = Path()
            path.move(to: CGPoint(x: kf, y: off))
            path.addLine(to: CGPoint(x: kf, y: end))
            ctx.stroke(path, with: .color(.black), lineWidth: 1)
        }

        // Hoshi marks
        for (hx, hy) in model.hoshiPoints() {
            let cx = off + sp * CGFloat(hx)
            let cy = off + sp * CGFloat(hy)
            let r = sp * 0.1
            let rect = CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(.black))
        }

        // Coordinate labels
        for k in 0..<n {
            let letter = String(goLetters[k])
            let number = "\(n - k)"
            let kf = off + sp * CGFloat(k)
            let fontSize = max(9, sp * 0.4)
            let font = Font.system(size: fontSize)

            // Letters top & bottom
            ctx.draw(Text(letter).font(font).foregroundColor(.black),
                     at: CGPoint(x: kf, y: off / 2), anchor: .center)
            ctx.draw(Text(letter).font(font).foregroundColor(.black),
                     at: CGPoint(x: kf, y: end + off / 2), anchor: .center)

            // Numbers left & right
            ctx.draw(Text(number).font(font).foregroundColor(.black),
                     at: CGPoint(x: off / 2, y: kf), anchor: .center)
            ctx.draw(Text(number).font(font).foregroundColor(.black),
                     at: CGPoint(x: end + off / 2, y: kf), anchor: .center)
        }

        // Stones
        let stoneRadius = (sp + 1) / 2
        for vertex in model.blackStones {
            drawDisc(ctx: ctx, vertex: vertex, sp: sp, off: off,
                     radius: stoneRadius, fill: .black, stroke: nil)
        }
        for vertex in model.whiteStones {
            drawDisc(ctx: ctx, vertex: vertex, sp: sp, off: off,
                     radius: stoneRadius, fill: .white, stroke: .black)
        }

        // Markups
        for markup in model.markups {
            drawMarkup(ctx: ctx, markup: markup, sp: sp, off: off)
        }
    }

    private func drawDisc(ctx: GraphicsContext, vertex: String, sp: CGFloat, off: CGFloat,
                          radius: CGFloat, fill: Color, stroke: Color?) {
        guard let (x, y) = model.vertexToXY(vertex) else { return }
        let cx = off + sp * CGFloat(x)
        let cy = off + sp * CGFloat(y)
        let rect = CGRect(x: cx - radius, y: cy - radius, width: radius * 2, height: radius * 2)
        ctx.fill(Path(ellipseIn: rect), with: .color(fill))
        if let s = stroke {
            ctx.stroke(Path(ellipseIn: rect), with: .color(s), lineWidth: 1)
        }
    }

    private func drawMarkup(ctx: GraphicsContext, markup: Markup, sp: CGFloat, off: CGFloat) {
        guard let (x, y) = model.vertexToXY(markup.vertex) else { return }
        let cx = off + sp * CGFloat(x)
        let cy = off + sp * CGFloat(y)
        let color = markup.color

        switch markup.symbol {
        case .circle:
            let r = sp / 3
            let rect = CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)
            ctx.stroke(Path(ellipseIn: rect), with: .color(color), lineWidth: 1.5)

        case .square:
            let d = sp / 4
            let rect = CGRect(x: cx - d, y: cy - d, width: d * 2, height: d * 2)
            ctx.stroke(Path(rect), with: .color(color), lineWidth: 1.5)

        case .bigSquare:
            let d = sp / 2 - 1
            let rect = CGRect(x: cx - d, y: cy - d, width: d * 2, height: d * 2)
            ctx.stroke(Path(rect), with: .color(color), lineWidth: 1.5)

        case .triangle:
            let d = sp / 2 - 1
            var path = Path()
            path.move(to: CGPoint(x: cx - d, y: cy + d))
            path.addLine(to: CGPoint(x: cx + d, y: cy + d))
            path.addLine(to: CGPoint(x: cx, y: cy - d))
            path.closeSubpath()
            ctx.stroke(path, with: .color(color), lineWidth: 1.5)

        case .dot:
            let r = sp / 6
            let rect = CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color))

        case .bigDot:
            let r = sp / 4
            let rect = CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color))

        case .smallDot:
            let r = sp / 9
            let rect = CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color))

        case .stone:
            let r = sp / 2 * 0.5
            let rect = CGRect(x: cx - r, y: cy - r, width: r * 2, height: r * 2)
            ctx.fill(Path(ellipseIn: rect), with: .color(color))

        case .text(let s):
            let fontSize = max(8, sp * 0.4)
            let t = Text(s).font(.system(size: fontSize)).foregroundStyle(color)
            ctx.draw(t, at: CGPoint(x: cx, y: cy), anchor: .center)
        }
    }
}

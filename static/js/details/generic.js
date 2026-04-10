// generic.js — Generic op detail visualization

export function drawGenericDetail(svg, op) {
    const g = svg.append('g').attr('transform', 'translate(20, 30)');
    let y = 0;

    g.append('text').attr('class', 'tensor-label')
        .attr('x', 0).attr('y', y).text(`Op: ${op.label}`);
    y += 25;

    g.append('text').attr('class', 'dim-label').attr('fill', '#aaa')
        .attr('x', 0).attr('y', y).text(`Type: ${op.type}`);
    y += 20;

    g.append('text').attr('class', 'dim-label').attr('fill', '#aaa')
        .attr('x', 0).attr('y', y).text(`Inputs: ${op.inputs.join(', ')}`);
    y += 20;

    g.append('text').attr('class', 'dim-label').attr('fill', '#aaa')
        .attr('x', 0).attr('y', y).text(`Output: ${op.output}`);

    svg.attr('height', y + 30);
}

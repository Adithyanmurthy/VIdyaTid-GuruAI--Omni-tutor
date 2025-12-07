/**
 * Animated Globe with India Map
 * Creates a beautiful rotating globe with India highlighted
 */

document.addEventListener('DOMContentLoaded', function() {
    const heroAnimation = document.querySelector('.hero-animation');
    if (!heroAnimation) return;
    
    // Create SVG namespace
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = heroAnimation.querySelector('.animated-svg');
    if (!svg) return;
    
    // Create defs for gradients
    const defs = document.createElementNS(svgNS, 'defs');
    
    // Globe gradient
    const globeGradient = document.createElementNS(svgNS, 'radialGradient');
    globeGradient.setAttribute('id', 'globeGradient');
    globeGradient.setAttribute('cx', '30%');
    globeGradient.setAttribute('cy', '30%');
    globeGradient.innerHTML = `
        <stop offset="0%" style="stop-color:rgba(59, 130, 246, 0.15);stop-opacity:1" />
        <stop offset="100%" style="stop-color:rgba(59, 130, 246, 0.05);stop-opacity:1" />
    `;
    defs.appendChild(globeGradient);
    
    // India tricolor gradient
    const indiaGradient = document.createElementNS(svgNS, 'linearGradient');
    indiaGradient.setAttribute('id', 'indiaGradient');
    indiaGradient.setAttribute('x1', '0%');
    indiaGradient.setAttribute('y1', '0%');
    indiaGradient.setAttribute('x2', '0%');
    indiaGradient.setAttribute('y2', '100%');
    indiaGradient.innerHTML = `
        <stop offset="0%" style="stop-color:#ff9933;stop-opacity:1" />
        <stop offset="50%" style="stop-color:#ffffff;stop-opacity:0.9" />
        <stop offset="100%" style="stop-color:#138808;stop-opacity:1" />
    `;
    defs.appendChild(indiaGradient);
    
    svg.insertBefore(defs, svg.firstChild);
    
    // Create globe group
    const globeGroup = document.createElementNS(svgNS, 'g');
    globeGroup.setAttribute('class', 'globe-group');
    globeGroup.setAttribute('transform', 'translate(950, 200)');
    
    // Globe background
    const globeBg = document.createElementNS(svgNS, 'circle');
    globeBg.setAttribute('class', 'globe-bg');
    globeBg.setAttribute('cx', '0');
    globeBg.setAttribute('cy', '0');
    globeBg.setAttribute('r', '100');
    globeBg.setAttribute('fill', 'url(#globeGradient)');
    globeGroup.appendChild(globeBg);
    
    // Globe circle
    const globeCircle = document.createElementNS(svgNS, 'circle');
    globeCircle.setAttribute('class', 'globe-circle');
    globeCircle.setAttribute('cx', '0');
    globeCircle.setAttribute('cy', '0');
    globeCircle.setAttribute('r', '100');
    globeCircle.setAttribute('fill', 'none');
    globeCircle.setAttribute('stroke', '#3b82f6');
    globeCircle.setAttribute('stroke-width', '2.5');
    globeCircle.setAttribute('opacity', '0.4');
    globeGroup.appendChild(globeCircle);
    
    // Add latitude lines
    const latitudes = [
        {cy: -50, ry: 15},
        {cy: -25, ry: 20},
        {cy: 0, ry: 25},
        {cy: 25, ry: 20},
        {cy: 50, ry: 15}
    ];
    
    latitudes.forEach((lat, index) => {
        const ellipse = document.createElementNS(svgNS, 'ellipse');
        ellipse.setAttribute('class', `lat-line lat-${index + 1}`);
        ellipse.setAttribute('cx', '0');
        ellipse.setAttribute('cy', lat.cy);
        ellipse.setAttribute('rx', '100');
        ellipse.setAttribute('ry', lat.ry);
        ellipse.setAttribute('fill', 'none');
        ellipse.setAttribute('stroke', index === 2 ? '#8b5cf6' : '#3b82f6');
        ellipse.setAttribute('stroke-width', index === 2 ? '1' : '0.8');
        ellipse.setAttribute('opacity', index === 2 ? '0.3' : '0.25');
        globeGroup.appendChild(ellipse);
    });
    
    // Add longitude lines
    const longitudes = [25, 50, 75];
    longitudes.forEach((rx, index) => {
        const ellipse = document.createElementNS(svgNS, 'ellipse');
        ellipse.setAttribute('class', `long-line long-${index + 1}`);
        ellipse.setAttribute('cx', '0');
        ellipse.setAttribute('cy', '0');
        ellipse.setAttribute('rx', rx);
        ellipse.setAttribute('ry', '100');
        ellipse.setAttribute('fill', 'none');
        ellipse.setAttribute('stroke', rx === 50 ? '#8b5cf6' : '#3b82f6');
        ellipse.setAttribute('stroke-width', rx === 50 ? '1' : '0.8');
        ellipse.setAttribute('opacity', rx === 50 ? '0.3' : '0.25');
        globeGroup.appendChild(ellipse);
    });
    
    // Central longitude line
    const centralLine = document.createElementNS(svgNS, 'line');
    centralLine.setAttribute('class', 'long-line long-4');
    centralLine.setAttribute('x1', '0');
    centralLine.setAttribute('y1', '-100');
    centralLine.setAttribute('x2', '0');
    centralLine.setAttribute('y2', '100');
    centralLine.setAttribute('stroke', '#8b5cf6');
    centralLine.setAttribute('stroke-width', '1');
    centralLine.setAttribute('opacity', '0.3');
    globeGroup.appendChild(centralLine);
    
    // India map group
    const indiaGroup = document.createElementNS(svgNS, 'g');
    indiaGroup.setAttribute('class', 'india-map-group');
    indiaGroup.setAttribute('transform', 'translate(15, -15) scale(1.2)');
    
    // Simplified but accurate India map path
    const indiaPath = document.createElementNS(svgNS, 'path');
    indiaPath.setAttribute('class', 'india-map');
    indiaPath.setAttribute('d', 'M 8,-35 L 12,-32 L 15,-28 L 18,-24 L 20,-20 L 22,-16 L 23,-12 L 24,-8 L 24,-4 L 24,0 L 23,4 L 22,8 L 20,12 L 18,16 L 16,20 L 14,24 L 12,27 L 10,29 L 8,31 L 6,32 L 4,33 L 2,33 L 0,33 L -2,32 L -4,31 L -6,30 L -8,28 L -10,26 L -12,24 L -14,22 L -16,20 L -18,18 L -20,16 L -22,14 L -23,12 L -24,10 L -24,8 L -23,6 L -22,4 L -21,2 L -20,0 L -19,-2 L -18,-4 L -17,-6 L -16,-8 L -15,-10 L -14,-12 L -13,-14 L -12,-16 L -11,-18 L -10,-20 L -9,-22 L -8,-24 L -7,-26 L -6,-28 L -5,-30 L -4,-32 L -3,-33 L -2,-34 L -1,-35 L 0,-35 L 2,-35 L 4,-35 L 6,-35 Z');
    indiaPath.setAttribute('fill', 'url(#indiaGradient)');
    indiaPath.setAttribute('opacity', '0.95');
    indiaPath.setAttribute('stroke', '#ff9933');
    indiaPath.setAttribute('stroke-width', '2');
    indiaPath.setAttribute('stroke-linejoin', 'round');
    indiaPath.setAttribute('stroke-linecap', 'round');
    indiaGroup.appendChild(indiaPath);
    
    // Add Andaman & Nicobar islands (Bay of Bengal - East side)
    const andamanIslands = [
        {cx: 24, cy: 8, r: 1.2, name: 'North Andaman'},
        {cx: 25, cy: 12, r: 1.5, name: 'Middle Andaman'},
        {cx: 26, cy: 16, r: 1.3, name: 'South Andaman'},
        {cx: 27, cy: 20, r: 1, name: 'Little Andaman'},
        {cx: 28, cy: 25, r: 0.8, name: 'Car Nicobar'},
        {cx: 29, cy: 28, r: 1.1, name: 'Great Nicobar'}
    ];
    
    andamanIslands.forEach(island => {
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('class', 'andaman');
        circle.setAttribute('cx', island.cx);
        circle.setAttribute('cy', island.cy);
        circle.setAttribute('r', island.r);
        circle.setAttribute('fill', '#138808');
        circle.setAttribute('opacity', '0.75');
        circle.setAttribute('stroke', '#ff9933');
        circle.setAttribute('stroke-width', '0.5');
        indiaGroup.appendChild(circle);
    });
    
    // Add Lakshadweep islands (Arabian Sea - West side)
    const lakshadweepIslands = [
        {cx: -22, cy: 5, r: 0.6},
        {cx: -23, cy: 7, r: 0.7},
        {cx: -22.5, cy: 9, r: 0.5},
        {cx: -23.5, cy: 11, r: 0.6},
        {cx: -24, cy: 13, r: 0.5}
    ];
    
    lakshadweepIslands.forEach(island => {
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('class', 'lakshadweep');
        circle.setAttribute('cx', island.cx);
        circle.setAttribute('cy', island.cy);
        circle.setAttribute('r', island.r);
        circle.setAttribute('fill', '#138808');
        circle.setAttribute('opacity', '0.75');
        circle.setAttribute('stroke', '#ff9933');
        circle.setAttribute('stroke-width', '0.5');
        indiaGroup.appendChild(circle);
    });
    
    globeGroup.appendChild(indiaGroup);
    
    // Capital marker (Delhi - positioned in North-Central India)
    const capitalGroup = document.createElementNS(svgNS, 'g');
    capitalGroup.setAttribute('class', 'capital-marker');
    capitalGroup.setAttribute('transform', 'translate(5, -22)');
    
    const capitalDot = document.createElementNS(svgNS, 'circle');
    capitalDot.setAttribute('class', 'capital-dot');
    capitalDot.setAttribute('cx', '0');
    capitalDot.setAttribute('cy', '0');
    capitalDot.setAttribute('r', '2.5');
    capitalDot.setAttribute('fill', '#fbbf24');
    capitalGroup.appendChild(capitalDot);
    
    const capitalPulse1 = document.createElementNS(svgNS, 'circle');
    capitalPulse1.setAttribute('class', 'capital-pulse-1');
    capitalPulse1.setAttribute('cx', '0');
    capitalPulse1.setAttribute('cy', '0');
    capitalPulse1.setAttribute('r', '2.5');
    capitalPulse1.setAttribute('fill', 'none');
    capitalPulse1.setAttribute('stroke', '#fbbf24');
    capitalPulse1.setAttribute('stroke-width', '2');
    capitalGroup.appendChild(capitalPulse1);
    
    const capitalPulse2 = document.createElementNS(svgNS, 'circle');
    capitalPulse2.setAttribute('class', 'capital-pulse-2');
    capitalPulse2.setAttribute('cx', '0');
    capitalPulse2.setAttribute('cy', '0');
    capitalPulse2.setAttribute('r', '2.5');
    capitalPulse2.setAttribute('fill', 'none');
    capitalPulse2.setAttribute('stroke', '#ff9933');
    capitalPulse2.setAttribute('stroke-width', '1.5');
    capitalGroup.appendChild(capitalPulse2);
    
    globeGroup.appendChild(capitalGroup);
    
    // Add connection lines
    const connections = [
        {x1: 15, y1: -15, x2: 60, y2: -60, stroke: '#3b82f6'},
        {x1: 15, y1: 0, x2: 80, y2: 0, stroke: '#8b5cf6'},
        {x1: 15, y1: 15, x2: 60, y2: 60, stroke: '#06b6d4'}
    ];
    
    connections.forEach((conn, index) => {
        const line = document.createElementNS(svgNS, 'line');
        line.setAttribute('class', `connection-line line-${index + 1}`);
        line.setAttribute('x1', conn.x1);
        line.setAttribute('y1', conn.y1);
        line.setAttribute('x2', conn.x2);
        line.setAttribute('y2', conn.y2);
        line.setAttribute('stroke', conn.stroke);
        line.setAttribute('stroke-width', '1');
        line.setAttribute('opacity', '0.3');
        globeGroup.appendChild(line);
    });
    
    // Add orbiting satellites
    const satellites = [
        {cx: 0, cy: -120, r: 3, fill: '#fbbf24', class: 'sat-1'},
        {cx: 85, cy: 85, r: 2.5, fill: '#3b82f6', class: 'sat-2'},
        {cx: -85, cy: 85, r: 2.5, fill: '#8b5cf6', class: 'sat-3'}
    ];
    
    satellites.forEach(sat => {
        const circle = document.createElementNS(svgNS, 'circle');
        circle.setAttribute('class', `satellite ${sat.class}`);
        circle.setAttribute('cx', sat.cx);
        circle.setAttribute('cy', sat.cy);
        circle.setAttribute('r', sat.r);
        circle.setAttribute('fill', sat.fill);
        globeGroup.appendChild(circle);
    });
    
    // Insert globe group at the beginning of SVG
    svg.insertBefore(globeGroup, svg.firstChild.nextSibling);
});

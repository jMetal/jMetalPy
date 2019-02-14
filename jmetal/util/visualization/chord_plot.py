import colorsys
from typing import List

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.path import Path
from tqdm import tqdm

from jmetal.core.solution import FloatSolution


def polar_to_cartesian(r, theta):
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def draw_sector(startAngle=0, endAngle=60, radius=1.0, width=0.2, lw=2, ls='-', ax=None, fc=(1, 0, 0), ec=(0, 0, 0),
                z_order=1):
    if startAngle > endAngle:
        startAngle, endAngle = endAngle, startAngle
    startAngle *= np.pi / 180.
    endAngle *= np.pi / 180.

    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4. / 3. * np.tan((endAngle - startAngle) / 4.) * radius
    inner = radius * (1 - width)

    vertsPath = [polar_to_cartesian(radius, startAngle),
                 polar_to_cartesian(radius, startAngle) + polar_to_cartesian(opt, startAngle + 0.5 * np.pi),
                 polar_to_cartesian(radius, endAngle) + polar_to_cartesian(opt, endAngle - 0.5 * np.pi),
                 polar_to_cartesian(radius, endAngle),
                 polar_to_cartesian(inner, endAngle),
                 polar_to_cartesian(inner, endAngle) + polar_to_cartesian(opt * (1 - width), endAngle - 0.5 * np.pi),
                 polar_to_cartesian(inner, startAngle) + polar_to_cartesian(opt * (1 - width),
                                                                            startAngle + 0.5 * np.pi),
                 polar_to_cartesian(inner, startAngle),
                 polar_to_cartesian(radius, startAngle)]

    codesPaths = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
                  Path.CURVE4, Path.CLOSEPOLY]

    if ax is None:
        return vertsPath, codesPaths
    else:
        path = path.Path(vertsPath, codesPaths)
        patch = patches.PathPatch(path, facecolor=fc, edgecolor=ec, lw=lw, linestyle=ls, zorder=z_order)
        ax.add_patch(patch)
        return (patch)


def draw_chord(startAngle1=0, endAngle1=60, startAngle2=180, endAngle2=240, radius=1.0, chordwidth=0.7, ax=None,
               color=(1, 0, 0), z_order=1):
    if startAngle1 > endAngle1:
        startAngle1, endAngle1 = endAngle1, startAngle1
    if startAngle2 > endAngle2:
        startAngle2, endAngle2 = endAngle2, startAngle2
    startAngle1 *= np.pi / 180.
    endAngle1 *= np.pi / 180.
    startAngle2 *= np.pi / 180.
    endAngle2 *= np.pi / 180.

    optAngle1 = 4. / 3. * np.tan((endAngle1 - startAngle1) / 4.) * radius
    optAngle2 = 4. / 3. * np.tan((endAngle2 - startAngle2) / 4.) * radius
    rchord = radius * (1 - chordwidth)

    vertsPath = [polar_to_cartesian(radius, startAngle1),
                 polar_to_cartesian(radius, startAngle1) + polar_to_cartesian(optAngle1, startAngle1 + 0.5 * np.pi),
                 polar_to_cartesian(radius, endAngle1) + polar_to_cartesian(optAngle1, endAngle1 - 0.5 * np.pi),
                 polar_to_cartesian(radius, endAngle1),
                 polar_to_cartesian(rchord, endAngle1), polar_to_cartesian(rchord, startAngle2),
                 polar_to_cartesian(radius, startAngle2),
                 polar_to_cartesian(radius, startAngle2) + polar_to_cartesian(optAngle2, startAngle2 + 0.5 * np.pi),
                 polar_to_cartesian(radius, endAngle2) + polar_to_cartesian(optAngle2, endAngle2 - 0.5 * np.pi),
                 polar_to_cartesian(radius, endAngle2),
                 polar_to_cartesian(rchord, endAngle2), polar_to_cartesian(rchord, startAngle1),
                 polar_to_cartesian(radius, startAngle1)]

    codesPath = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    if ax == None:
        return vertsPath, codesPath
    else:
        path = Path(vertsPath, codesPath)
        patch = patches.PathPatch(path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=2, alpha=0.5)
        ax.add_patch(patch)
        return (patch)


def hover_over_bin(event, handleTickers, handlePlots, colors, fig):
    flagFound = False
    for iobj in range(len(handleTickers)):
        for ibin in range(len(handleTickers[iobj])):
            cont = False
            if flagFound == False:
                cont, ind = handleTickers[iobj][ibin].contains(event)
                if cont == True:
                    flagFound = True
            if cont:
                plt.setp(handleTickers[iobj][ibin], facecolor=colors[iobj])
                [h.set_visible(True) for h in handlePlots[iobj][ibin]]
                flagFound = True
                fig.canvas.draw_idle()
            else:
                plt.setp(handleTickers[iobj][ibin], facecolor=(1, 1, 1))
                for h in handlePlots[iobj][ibin]:
                    h.set_visible(False)
                fig.canvas.draw_idle()


def chord_diagram(solutions: List[FloatSolution], nbins='auto', ax=None, labelsObj=None,
                  propLabels=dict(fontsize=13, ha='center', va='center'), pad=6):
    pointsMatrix = np.array([s.objectives for s in solutions])
    (NPOINTS, NOBJ) = np.shape(pointsMatrix)

    HSV_tuples = [(x * 1.0 / NOBJ, 0.5, 0.5) for x in range(NOBJ)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes([0, 0, 1, 1], aspect='equal')

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.axis('off')

    y = np.array([1. / NOBJ] * NOBJ) * (360 - pad * NOBJ)
    sectorAngles = []
    labelsPosandRos = []

    startAngle = 0
    for i in range(NOBJ):
        endAngle = startAngle + y[i]
        sectorAngles.append((startAngle, endAngle))
        angleDiff = 0.5 * (startAngle + endAngle)
        if -30 <= angleDiff <= 210:
            angleDiff -= 90
        else:
            angleDiff -= 270
        angleText = startAngle - 2.5
        if -30 <= angleText <= 210:
            angleText -= 90
        else:
            angleText -= 270

        labelsPosandRos.append(
            tuple(polar_to_cartesian(1.0, 0.5 * (startAngle + endAngle) * np.pi / 180.)) + (angleDiff,) +
            tuple(polar_to_cartesian(0.725, (startAngle - 2.5) * np.pi / 180.)) + (angleText,) +
            tuple(polar_to_cartesian(0.85, (startAngle - 2.5) * np.pi / 180.)) + (angleText,))
        startAngle = endAngle + pad

    arcPoints = []
    for point in pointsMatrix:
        arcPoints.append([])
        idim = 0
        for dim in point:
            anglePoint = sectorAngles[idim][0] + (sectorAngles[idim][1] - sectorAngles[idim][0]) * point[idim]
            arcPoints[-1].append((anglePoint, anglePoint))
            idim = idim + 1

    maxHistValues = []
    handleTickers = []
    handlePlots = []

    for iobj in tqdm(range(NOBJ), ascii=True, desc='Chord diagram'):

        draw_sector(startAngle=sectorAngles[iobj][0], endAngle=sectorAngles[iobj][1], radius=0.925, width=0.225, ax=ax,
                    fc=(1, 1, 1, 0.0), ec=(0, 0, 0), lw=2, z_order=10)
        draw_sector(startAngle=sectorAngles[iobj][0], endAngle=sectorAngles[iobj][1], radius=0.925, width=0.05, ax=ax,
                    fc=colors[iobj], ec=(0, 0, 0), lw=2, z_order=10)
        draw_sector(startAngle=sectorAngles[iobj][0], endAngle=sectorAngles[iobj][1], radius=0.7 + 0.15, width=0.0,
                    ax=ax, fc=colors[iobj], ec=colors[iobj], lw=2, ls=':', z_order=5)

        histValues, binsDim = np.histogram(pointsMatrix[:, iobj], bins=nbins)
        relativeHeightBinPre = 0.025
        maxHistValues.append(max(histValues))
        handleTickers.append([])
        handlePlots.append([])

        for indexBin in range(len(histValues)):

            startAngleBin = sectorAngles[iobj][0] + (sectorAngles[iobj][1] - sectorAngles[iobj][0]) * binsDim[indexBin]
            endAngleBin = sectorAngles[iobj][0] + (sectorAngles[iobj][1] - sectorAngles[iobj][0]) * binsDim[
                indexBin + 1]
            relativeHeightBin = 0.15 * histValues[indexBin] / max(histValues)
            handleTickers[-1].append(
                draw_sector(startAngle=startAngleBin, endAngle=endAngleBin, radius=0.69, width=0.08, ax=ax, lw=1,
                            fc=(1, 1, 1), ec=(0, 0, 0)))
            handlePlots[-1].append([])

            if histValues[indexBin] > 0:
                draw_sector(startAngle=startAngleBin, endAngle=endAngleBin, radius=0.7 + relativeHeightBin, width=0,
                            ax=ax, lw=1, fc=colors[iobj], ec=colors[iobj])
                plotPoint1 = polar_to_cartesian(0.7 + relativeHeightBinPre, startAngleBin * np.pi / 180.)
                plotPoint2 = polar_to_cartesian(0.7 + relativeHeightBin, startAngleBin * np.pi / 180.)
                plt.plot([plotPoint1[0], plotPoint2[0]], [plotPoint1[1], plotPoint2[1]], c=colors[iobj], lw=1)
                relativeHeightBinPre = relativeHeightBin
            else:
                plotPoint1 = polar_to_cartesian(0.7 + relativeHeightBinPre, startAngleBin * np.pi / 180.)
                plotPoint2 = polar_to_cartesian(0.725 + relativeHeightBin, startAngleBin * np.pi / 180.)
                plt.plot([plotPoint1[0], plotPoint2[0]], [plotPoint1[1], plotPoint2[1]], c=colors[iobj], lw=1)
                relativeHeightBinPre = 0.025
            if indexBin == len(histValues) - 1:
                plotPoint1 = polar_to_cartesian(0.7 + relativeHeightBin, endAngleBin * np.pi / 180.)
                plotPoint2 = polar_to_cartesian(0.725, endAngleBin * np.pi / 180.)
                plt.plot([plotPoint1[0], plotPoint2[0]], [plotPoint1[1], plotPoint2[1]], c=colors[iobj], lw=1)

            for ipoint in range(len(pointsMatrix)):
                plotPoint1 = polar_to_cartesian(0.6, arcPoints[ipoint][iobj][0] * np.pi / 180.)
                plotPoint2 = polar_to_cartesian(0.6, arcPoints[ipoint][iobj][0] * np.pi / 180.)
                plt.plot([plotPoint1[0], plotPoint2[0]], [plotPoint1[1], plotPoint2[1]], marker='o', markersize=3,
                         c=colors[iobj], lw=2)

                if binsDim[indexBin] < pointsMatrix[ipoint, iobj] <= binsDim[
                    indexBin + 1]:
                    for jdim in range(NOBJ):
                        if jdim >= 1:
                            handlePlots[iobj][indexBin].append(
                                draw_chord(arcPoints[ipoint][jdim - 1][0], arcPoints[ipoint][jdim - 1][1],
                                           arcPoints[ipoint][jdim][0], arcPoints[ipoint][jdim][1], radius=0.55,
                                           color=colors[iobj], chordwidth=1, ax=ax))
                            handlePlots[iobj][indexBin][-1].set_visible(False)
                    handlePlots[iobj][indexBin].append(
                        draw_chord(arcPoints[ipoint][-1][0], arcPoints[ipoint][-1][1], arcPoints[ipoint][0][0],
                                   arcPoints[ipoint][0][1], radius=0.55, color=colors[iobj], chordwidth=1, ax=ax))
                    handlePlots[iobj][indexBin][-1].set_visible(False)

    if labelsObj is None:
        labelsObj = ['$f_{' + str(i) + '}(\mathbf{x})$' for i in range(NOBJ)]

    propLegendBins = dict(fontsize=9, ha='center', va='center')

    for i in range(NOBJ):
        p0, p1 = polar_to_cartesian(0.975, sectorAngles[i][0] * np.pi / 180.)
        ax.text(p0, p1, '0', **propLegendBins)
        p0, p1 = polar_to_cartesian(0.975, sectorAngles[i][1] * np.pi / 180.)
        ax.text(p0, p1, '1', **propLegendBins)
        ax.text(labelsPosandRos[i][0], labelsPosandRos[i][1], labelsObj[i], rotation=labelsPosandRos[i][2],
                **propLabels)
        ax.text(labelsPosandRos[i][3], labelsPosandRos[i][4], '0', **propLegendBins, color=colors[i])
        ax.text(labelsPosandRos[i][6], labelsPosandRos[i][7], str(maxHistValues[i]), **propLegendBins, color=colors[i])

    plt.axis([-1.2, 1.2, -1.2, 1.2])
    fig.canvas.mpl_connect("motion_notify_event",
                           lambda event: hover_over_bin(event, handleTickers, handlePlots, colors, fig))
    plt.show()

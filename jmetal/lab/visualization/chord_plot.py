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


def draw_sector(start_angle=0, end_angle=60, radius=1.0, width=0.2, lw=2, ls='-', ax=None, fc=(1, 0, 0), ec=(0, 0, 0),
                z_order=1):
    if start_angle > end_angle:
        start_angle, end_angle = end_angle, start_angle
    start_angle *= np.pi / 180.
    end_angle *= np.pi / 180.

    # https://stackoverflow.com/questions/1734745/how-to-create-circle-with-b%C3%A9zier-curves
    opt = 4. / 3. * np.tan((end_angle - start_angle) / 4.) * radius
    inner = radius * (1 - width)

    vertsPath = [polar_to_cartesian(radius, start_angle),
                 polar_to_cartesian(radius, start_angle) + polar_to_cartesian(opt, start_angle + 0.5 * np.pi),
                 polar_to_cartesian(radius, end_angle) + polar_to_cartesian(opt, end_angle - 0.5 * np.pi),
                 polar_to_cartesian(radius, end_angle),
                 polar_to_cartesian(inner, end_angle),
                 polar_to_cartesian(inner, end_angle) + polar_to_cartesian(opt * (1 - width), end_angle - 0.5 * np.pi),
                 polar_to_cartesian(inner, start_angle) + polar_to_cartesian(opt * (1 - width),
                                                                             start_angle + 0.5 * np.pi),
                 polar_to_cartesian(inner, start_angle),
                 polar_to_cartesian(radius, start_angle)]

    codesPaths = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
                  Path.CURVE4, Path.CLOSEPOLY]

    if ax is None:
        return vertsPath, codesPaths
    else:
        path = Path(vertsPath, codesPaths)
        patch = patches.PathPatch(path, facecolor=fc, edgecolor=ec, lw=lw, linestyle=ls, zorder=z_order)
        ax.add_patch(patch)
        return (patch)


def draw_chord(start_angle1=0, end_angle1=60, start_angle2=180, end_angle2=240, radius=1.0, chord_width=0.7, ax=None,
               color=(1, 0, 0), z_order=1):
    if start_angle1 > end_angle1:
        start_angle1, end_angle1 = end_angle1, start_angle1
    if start_angle2 > end_angle2:
        start_angle2, end_angle2 = end_angle2, start_angle2
    start_angle1 *= np.pi / 180.
    end_angle1 *= np.pi / 180.
    start_angle2 *= np.pi / 180.
    end_angle2 *= np.pi / 180.

    optAngle1 = 4. / 3. * np.tan((end_angle1 - start_angle1) / 4.) * radius
    optAngle2 = 4. / 3. * np.tan((end_angle2 - start_angle2) / 4.) * radius
    rchord = radius * (1 - chord_width)

    vertsPath = [polar_to_cartesian(radius, start_angle1),
                 polar_to_cartesian(radius, start_angle1) + polar_to_cartesian(optAngle1, start_angle1 + 0.5 * np.pi),
                 polar_to_cartesian(radius, end_angle1) + polar_to_cartesian(optAngle1, end_angle1 - 0.5 * np.pi),
                 polar_to_cartesian(radius, end_angle1),
                 polar_to_cartesian(rchord, end_angle1), polar_to_cartesian(rchord, start_angle2),
                 polar_to_cartesian(radius, start_angle2),
                 polar_to_cartesian(radius, start_angle2) + polar_to_cartesian(optAngle2, start_angle2 + 0.5 * np.pi),
                 polar_to_cartesian(radius, end_angle2) + polar_to_cartesian(optAngle2, end_angle2 - 0.5 * np.pi),
                 polar_to_cartesian(radius, end_angle2),
                 polar_to_cartesian(rchord, end_angle2), polar_to_cartesian(rchord, start_angle1),
                 polar_to_cartesian(radius, start_angle1)]

    codesPath = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    if ax == None:
        return vertsPath, codesPath
    else:
        path = Path(vertsPath, codesPath)
        patch = patches.PathPatch(path, facecolor=color + (0.5,), edgecolor=color + (0.4,), lw=2, alpha=0.5)
        ax.add_patch(patch)
        return (patch)


def hover_over_bin(event, handle_tickers, handle_plots, colors, fig):
    is_found = False

    for iobj in range(len(handle_tickers)):
        for ibin in range(len(handle_tickers[iobj])):
            cont = False
            if not is_found:
                cont, ind = handle_tickers[iobj][ibin].contains(event)
                if cont:
                    is_found = True
            if cont:
                plt.setp(handle_tickers[iobj][ibin], facecolor=colors[iobj])
                [h.set_visible(True) for h in handle_plots[iobj][ibin]]
                is_found = True
                fig.canvas.draw_idle()
            else:
                plt.setp(handle_tickers[iobj][ibin], facecolor=(1, 1, 1))
                for h in handle_plots[iobj][ibin]:
                    h.set_visible(False)
                fig.canvas.draw_idle()


def chord_diagram(solutions: List[FloatSolution], nbins='auto', ax=None, obj_labels=None,
                  prop_labels=dict(fontsize=13, ha='center', va='center'), pad=6):
    points_matrix = np.array([s.objectives for s in solutions])
    (NPOINTS, NOBJ) = np.shape(points_matrix)

    HSV_tuples = [(x * 1.0 / NOBJ, 0.5, 0.5) for x in range(NOBJ)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes([0, 0, 1, 1], aspect='equal')

    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.axis('off')

    y = np.array([1. / NOBJ] * NOBJ) * (360 - pad * NOBJ)
    sector_angles = []
    labels_pos_and_ros = []

    start_angle = 0
    for i in range(NOBJ):
        end_angle = start_angle + y[i]
        sector_angles.append((start_angle, end_angle))
        angle_diff = 0.5 * (start_angle + end_angle)
        if -30 <= angle_diff <= 210:
            angle_diff -= 90
        else:
            angle_diff -= 270
        angleText = start_angle - 2.5
        if -30 <= angleText <= 210:
            angleText -= 90
        else:
            angleText -= 270

        labels_pos_and_ros.append(
            tuple(polar_to_cartesian(1.0, 0.5 * (start_angle + end_angle) * np.pi / 180.)) + (angle_diff,) +
            tuple(polar_to_cartesian(0.725, (start_angle - 2.5) * np.pi / 180.)) + (angleText,) +
            tuple(polar_to_cartesian(0.85, (start_angle - 2.5) * np.pi / 180.)) + (angleText,))
        start_angle = end_angle + pad

    arc_points = []
    for point in points_matrix:
        arc_points.append([])
        idim = 0

        for _ in point:
            anglePoint = sector_angles[idim][0] + (sector_angles[idim][1] - sector_angles[idim][0]) * point[idim]
            arc_points[-1].append((anglePoint, anglePoint))
            idim = idim + 1

    max_hist_values = []
    handle_tickers = []
    handle_plots = []

    for iobj in tqdm(range(NOBJ), ascii=True, desc='Chord diagram'):
        draw_sector(start_angle=sector_angles[iobj][0], end_angle=sector_angles[iobj][1], radius=0.925, width=0.225,
                    ax=ax,
                    fc=(1, 1, 1, 0.0), ec=(0, 0, 0), lw=2, z_order=10)
        draw_sector(start_angle=sector_angles[iobj][0], end_angle=sector_angles[iobj][1], radius=0.925, width=0.05,
                    ax=ax,
                    fc=colors[iobj], ec=(0, 0, 0), lw=2, z_order=10)
        draw_sector(start_angle=sector_angles[iobj][0], end_angle=sector_angles[iobj][1], radius=0.7 + 0.15, width=0.0,
                    ax=ax, fc=colors[iobj], ec=colors[iobj], lw=2, ls=':', z_order=5)

        histValues, binsDim = np.histogram(points_matrix[:, iobj], bins=nbins)
        relativeHeightBinPre = 0.025
        max_hist_values.append(max(histValues))
        handle_tickers.append([])
        handle_plots.append([])

        for indexBin in range(len(histValues)):
            startAngleBin = sector_angles[iobj][0] + (sector_angles[iobj][1] - sector_angles[iobj][0]) * binsDim[
                indexBin]
            endAngleBin = sector_angles[iobj][0] + (sector_angles[iobj][1] - sector_angles[iobj][0]) * binsDim[
                indexBin + 1]
            relativeHeightBin = 0.15 * histValues[indexBin] / max(histValues)
            handle_tickers[-1].append(
                draw_sector(start_angle=startAngleBin, end_angle=endAngleBin, radius=0.69, width=0.08, ax=ax, lw=1,
                            fc=(1, 1, 1), ec=(0, 0, 0)))
            handle_plots[-1].append([])

            if histValues[indexBin] > 0:
                draw_sector(start_angle=startAngleBin, end_angle=endAngleBin, radius=0.7 + relativeHeightBin, width=0,
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

            for ipoint in range(len(points_matrix)):
                plotPoint1 = polar_to_cartesian(0.6, arc_points[ipoint][iobj][0] * np.pi / 180.)
                plotPoint2 = polar_to_cartesian(0.6, arc_points[ipoint][iobj][0] * np.pi / 180.)
                plt.plot([plotPoint1[0], plotPoint2[0]], [plotPoint1[1], plotPoint2[1]], marker='o', markersize=3,
                         c=colors[iobj], lw=2)

                if binsDim[indexBin] < points_matrix[ipoint, iobj] <= binsDim[
                    indexBin + 1]:
                    for jdim in range(NOBJ):
                        if jdim >= 1:
                            handle_plots[iobj][indexBin].append(
                                draw_chord(arc_points[ipoint][jdim - 1][0], arc_points[ipoint][jdim - 1][1],
                                           arc_points[ipoint][jdim][0], arc_points[ipoint][jdim][1], radius=0.55,
                                           color=colors[iobj], chord_width=1, ax=ax))
                            handle_plots[iobj][indexBin][-1].set_visible(False)
                    handle_plots[iobj][indexBin].append(
                        draw_chord(arc_points[ipoint][-1][0], arc_points[ipoint][-1][1], arc_points[ipoint][0][0],
                                   arc_points[ipoint][0][1], radius=0.55, color=colors[iobj], chord_width=1, ax=ax))
                    handle_plots[iobj][indexBin][-1].set_visible(False)

    if obj_labels is None:
        obj_labels = ['$f_{' + str(i) + '}(\mathbf{x})$' for i in range(NOBJ)]

    prop_legend_bins = dict(fontsize=9, ha='center', va='center')

    for i in range(NOBJ):
        p0, p1 = polar_to_cartesian(0.975, sector_angles[i][0] * np.pi / 180.)
        ax.text(p0, p1, '0', **prop_legend_bins)
        p0, p1 = polar_to_cartesian(0.975, sector_angles[i][1] * np.pi / 180.)
        ax.text(p0, p1, '1', **prop_legend_bins)
        ax.text(labels_pos_and_ros[i][0], labels_pos_and_ros[i][1], obj_labels[i], rotation=labels_pos_and_ros[i][2],
                **prop_labels)
        ax.text(labels_pos_and_ros[i][3], labels_pos_and_ros[i][4], '0', **prop_legend_bins, color=colors[i])
        ax.text(labels_pos_and_ros[i][6], labels_pos_and_ros[i][7], str(max_hist_values[i]), **prop_legend_bins,
                color=colors[i])

    plt.axis([-1.2, 1.2, -1.2, 1.2])
    fig.canvas.mpl_connect("motion_notify_event",
                           lambda event: hover_over_bin(event, handle_tickers, handle_plots, colors, fig))
    plt.show()

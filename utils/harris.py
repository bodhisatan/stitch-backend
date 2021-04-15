"Imports"
import numpy as np
from scipy.ndimage import filters
from PIL import Image
import utils.imutils
import matplotlib.pylab as pyl
import cv2

from utils.pic_analysis import pil_image_to_cv


def generateHarrisMatrix(image, sigma=2):  # sigma is always 2
    xOfI = np.zeros(image.shape)
    filters.gaussian_filter(image, 1, (0, 1), output=xOfI)
    yOfI = np.zeros(image.shape)
    filters.gaussian_filter(image, 1, (1, 0), output=yOfI)

    # Compute the components of the Harris matrix - used to find Trace and Determinant below
    A = filters.gaussian_filter(xOfI * xOfI, sigma)
    B = filters.gaussian_filter(xOfI * yOfI, sigma)
    C = filters.gaussian_filter(yOfI * yOfI, sigma)

    # Find the Trace and Detminant - used to calculate R
    determinantM = (A * C) - (B ** 2)
    traceM = (A + C)

    return determinantM / traceM


"Find Harris corner points above a threshold and perform nonmax suppression in the region +/- minimumDistance."


def findHarrisPoints(harrisImage, minimumDistance=10, threshold=0.1):
    # Find the top corner candidates above a threshold
    cornerThreshold = harrisImage.max() * threshold
    harrisImageThresholded = (harrisImage > cornerThreshold)

    # Find the co-ordinates of these candidates and their response values
    coordinates = np.array(harrisImageThresholded.nonzero()).T
    candidateValues = np.array([harrisImage[c[0], c[1]] for c in coordinates])

    # Find the indices in the candidateValues array that sort it in order of increasing response strength
    indices = np.argsort(candidateValues)

    # Store the allowed point locations in a Boolean Image
    allowedLocations = np.zeros(harrisImage.shape, dtype='bool')
    allowedLocations[minimumDistance:-minimumDistance, minimumDistance:-minimumDistance] = True

    # Select the best points using nonmax suppression based on the allowedLocations array
    filteredCoordinates = []
    for i in indices[::-1]:
        r, c = coordinates[i]
        if allowedLocations[r, c]:
            filteredCoordinates.append((r, c))
            allowedLocations[r - minimumDistance:r + minimumDistance, c - minimumDistance:c + minimumDistance] = False

    return filteredCoordinates


def plotHarrisInterestPoints(image, interestPoints):
    pyl.figure('Harris points/corners')
    pyl.imshow(image, cmap='gray')
    pyl.plot([p[1] for p in interestPoints], [p[0] for p in interestPoints], 'ro')
    pyl.axis('off')
    pyl.show()


def findDescriptors(image, interestPoints, width=5):
    descriptors = []
    for coords in interestPoints:
        patch = image[coords[0] - width:coords[0] + width + 1, coords[1] - width:coords[1] + width + 1].flatten()
        patch -= np.mean(patch)
        patch /= np.linalg.norm(patch)
        descriptors.append(patch)

    return descriptors


def matchDescriptors(descriptors1, descriptors2, threshold=0.95):
    array1 = np.asarray(descriptors1, dtype=np.float32)
    array2 = np.asarray(descriptors2, dtype=np.float32).T  # note the Transpose

    # Find the maximum values of array1, array2 and the dot product
    responseMatrix = np.dot(array1, array2)
    max1 = array1.max()
    max2 = array2.max()
    maxDotProduct = responseMatrix.max()

    # Initial, non-thresholded dot product - compared with the thresholded version below
    originalMatrix = Image.fromarray(responseMatrix * 255)

    pairs = []
    for r in range(responseMatrix.shape[0]):
        rowMaximum = responseMatrix[r][0]
        for c in range(responseMatrix.shape[1]):
            if (responseMatrix[r][c] > threshold) and (responseMatrix[r][c] > rowMaximum):
                pairs.append((r, c))
            else:
                responseMatrix[r][c] = 0

    # Compare the above matrix with the new, thresholded matrix    
    thresholdedMatrix = Image.fromarray(responseMatrix * 255)

    # In order: Maximum of array1, maximum of array2, maximum of Dot Product,
    # Image before thresholding, Image after thresholding and Pairs list
    return max1, max2, maxDotProduct, originalMatrix, thresholdedMatrix, pairs


def plotMatches(image1, image2, interestPoints1, interestPoints2, pairs):
    rows1 = image1.shape[0]
    rows2 = image2.shape[0]

    if rows1 < rows2:
        image1 = np.concatenate((image1, np.zeros((rows2 - rows1, image1.shape[1]))), axis=0)
    elif rows2 < rows1:
        image2 = np.concatenate((image2, np.zeros((rows1 - rows2, image2.shape[1]))), axis=0)

    # create new image with two input images appended side-by-side, then plot matches
    image3 = np.concatenate((image1, image2), axis=1)

    # note outliers in this image - RANSAC will remove these later
    # pyl.imshow(image3, cmap="gray")
    column1 = image1.shape[1]

    # plot each line using the indexes recovered from pairs
    for i in range(len(pairs)):
        index1, index2 = pairs[i]
        cv2.line(image3, (interestPoints1[index1][1], interestPoints1[index1][0]), (interestPoints2[index2][1] + column1, interestPoints2[index2][0]),(0, 255, 0), 1)
        # pyl.plot([interestPoints1[index1][1], interestPoints2[index2][1] + column1],
        #          [interestPoints1[index1][0], interestPoints2[index2][0]], 'c')

    # pyl.axis('off')
    # cv2.imwrite("aa.png", image3)
    # pyl.show()
    return image3


def RANSAC(matches, coordinates1, coordinates2, matchDistance=1.6):
    d2 = matchDistance ** 2
    ## Build a list of offsets from the lists of matching points for the
    ## first and second images.
    offsets = np.zeros((len(matches), 2))
    for i in range(len(matches)):
        index1, index2 = matches[i]
        offsets[i, 0] = coordinates1[index1][0] - coordinates2[index2][0]
        offsets[i, 1] = coordinates1[index1][1] - coordinates2[index2][1]

    ## Run the comparison.  best_match_count keeps track of the size of the
    ## largest consensus set, and (best_row_offset,best_col_offset) the
    ## current offset associated with the largest consensus set found so far.
    best_match_count = -1
    best_row_offset, best_col_offset = 1e6, 1e6
    for i in range(len(offsets)):
        match_count = 1.0
        offi0 = offsets[i, 0]
        offi1 = offsets[i, 1]
        ## Only continue into j loop looking for consensus if this point hasn't
        ## been found and folded into a consensus set earlier.  Just improves
        ## efficiency.
        if (offi0 - best_row_offset) ** 2 + (offi1 - best_col_offset) ** 2 >= d2:
            sum_row_offsets, sum_col_offsets = offi0, offi1
            for j in range(len(matches)):
                if j != i:
                    offj0 = offsets[j, 0]
                    offj1 = offsets[j, 1]
                    if (offi0 - offj0) ** 2 + (offi1 - offj1) ** 2 < d2:
                        sum_row_offsets += offj0
                        sum_col_offsets += offj1
                        match_count += 1.0
            if match_count >= best_match_count:
                best_row_offset = sum_row_offsets / match_count
                best_col_offset = sum_col_offsets / match_count
                best_match_count = match_count

    return best_row_offset, best_col_offset, best_match_count


def appendImages(image1, image2, rowOffset, columnOffset):
    # Convert floats to ints
    rowOffset = int(rowOffset)
    columnOffset = int(columnOffset)

    canvas = Image.new(image1.mode, (image1.width + abs(columnOffset), image1.width + abs(
        rowOffset)))  # create new 'canvas' image with calculated dimensions
    canvas.paste(image1, (0, canvas.height - image1.height))  # paste image1
    canvas.paste(image2, (columnOffset, canvas.height - image1.height + rowOffset))  # paste image2

    # plot final composite image
    # pyl.figure('Final Composite Image')
    # pyl.imshow(canvas)
    # pyl.axis('off')
    # pyl.show()
    # cv2.imwrite("result.png", pil_image_to_cv(canvas))

    return pil_image_to_cv(canvas)


# print("\nReading Images")
# harrisImage1 = (np.array(Image.open('C:\\Users\\Bodhisatan\\Desktop\\image\\11.png').convert('L'), dtype=np.float32))
# harrisImage2 = (np.array(Image.open('C:\\Users\\Bodhisatan\\Desktop\\image\\12.png').convert('L'), dtype=np.float32))
# print("OK - Printing Images:")
# imutils.imshow(harrisImage1)
# imutils.imshow(harrisImage2)
#
# print("\nFinding Harris Matrices")
# image1 = generateHarrisMatrix(harrisImage1, 2)
# image2 = generateHarrisMatrix(harrisImage2, 2)
# print("OK - Printing Harris Matrix:")
# imutils.imshow(image1)
# imutils.imshow(image2)
#
# print("\nFinding Interest Points for both images")
# interestPoints1 = findHarrisPoints(image1)
# interestPoints2 = findHarrisPoints(image2)
# print("Found " + str(len(interestPoints1)) + " interest points in image 1.")
# print("Found " + str(len(interestPoints2)) + " interest points in image 2.")
# plotHarrisInterestPoints(harrisImage1, interestPoints1)
# plotHarrisInterestPoints(harrisImage2, interestPoints2)
#
# print("Finding Normalised Image Patches (Image Descriptors) for both images")
# descriptors1 = findDescriptors(harrisImage1, interestPoints1)
# descriptors2 = findDescriptors(harrisImage2, interestPoints2)
# print("OK")
#
# print("\nFinding matches between Descriptors")
# maxOfImage1, maxOfImage2, maxOfDotProduct, originalMatrix, thresholdedMatrix, pairsList = matchDescriptors(descriptors1,
#                                                                                                            descriptors2)
# # Do we even need these three lines?
# print("Maximum of Image1: " + str(maxOfImage1))
# print("Maximum of Image2: " + str(maxOfImage2))
# print("Maximum of Dot Product: " + str(maxOfDotProduct))
#
# print("\nResponse matrix before and after thresholding: ")
# pyl.subplot(121)
# pyl.imshow(originalMatrix)
# pyl.subplot(122)
# pyl.imshow(thresholdedMatrix)
# pyl.show()
#
# print("\nPlot the matches between the two images:")
# result = plotMatches(harrisImage1, harrisImage2, interestPoints1, interestPoints2, pairsList)
#
# print("\nRANSAC Operation to clean up the noisy mapping above:")
# rowOffset, columnOffset, bestMatches = RANSAC(pairsList, interestPoints1, interestPoints2)
# print('Number of agreements (best match count): ' + str(bestMatches))
# print('Row offset: ' + str(rowOffset))
# print('Column offset: ' + str(columnOffset))
#
# print("\nFinal Image Reconstruction:")
# colourImage1 = Image.open('C:\\Users\\Bodhisatan\\Desktop\\image\\11.png')
# colourImage2 = Image.open('C:\\Users\\Bodhisatan\\Desktop\\image\\12.png')
# final = appendImages(colourImage1, colourImage2, rowOffset, columnOffset)

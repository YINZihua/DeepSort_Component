# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray (ratio-format tlbr)
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlbr, confidence, image_np):
        # ... (BEGIN) me adding
        feature = [1]
        self.height, self.width = image_np.shape
        self.tlbr = [tlbr[1]*self.width, tlbr[0]*self.height,
                     tlbr[3]*self.width, tlbr[2]*self.height]

        self.tlwh = np.asarray([self.tlbr[0], self.tlbr[1],
                                self.tlbr[2] - self.tlbr[0], # Width
                                self.tlbr[3] - self.tlbr[1]  # Height
                                ])
        # ... (END) me adding

        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
    
    def to_tlwh(self):
        return self.tlwh


    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        return self.tlbr

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

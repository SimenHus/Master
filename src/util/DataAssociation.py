
import cv2


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.structs import Frame
    from src.util import Geometry


class Extractor:
    orb = cv2.ORB.create()

    @classmethod
    def extract(clc, image, mask) -> tuple[list[cv2.KeyPoint], list[cv2.Mat]]:
        return clc.orb.detectAndCompute(image, mask)


class Matcher:
    THRESH_LOW: int = 50
    THRESH_HIGH: int = 100
    HISTO_LENGTH: int = 30
    
    def __init__(self, nnratio: float = 0.6, check_ori: bool = True) -> None:
        self.nnratio = nnratio
        self.check_orientation = check_ori

    def search_for_initialization(self, frame1: 'Frame', frame2: 'Frame', prev_matched: list['Geometry.Point2']) -> list[cv2.DMatch]:

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(frame1.descriptors, frame2.descriptors)
        # Return matches sorted by distance (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
        sorted_matches: list[cv2.Mat] = sorted(matches, key=lambda x: x.distance)

        result = []
        for match in sorted_matches:
            if self.THRESH_LOW < match.distance < self.THRESH_HIGH:
                result.append(match)
        
        return result
    


    #     for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    #     {
    #         cv::KeyPoint kp1 = F1.mvKeysUn[i1];
    #         int level1 = kp1.octave;
    #         if(level1>0)
    #             continue;

    #         vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

    #         if(vIndices2.empty())
    #             continue;

    #         cv::Mat d1 = F1.mDescriptors.row(i1);

    #         int bestDist = INT_MAX;
    #         int bestDist2 = INT_MAX;
    #         int bestIdx2 = -1;

    #         for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
    #         {
    #             size_t i2 = *vit;

    #             cv::Mat d2 = F2.mDescriptors.row(i2);

    #             int dist = DescriptorDistance(d1,d2);

    #             if(vMatchedDistance[i2]<=dist)
    #                 continue;

    #             if(dist<bestDist)
    #             {
    #                 bestDist2=bestDist;
    #                 bestDist=dist;
    #                 bestIdx2=i2;
    #             }
    #             else if(dist<bestDist2)
    #             {
    #                 bestDist2=dist;
    #             }
    #         }

    #         if(bestDist<=TH_LOW)
    #         {
    #             if(bestDist<(float)bestDist2*mfNNratio)
    #             {
    #                 if(vnMatches21[bestIdx2]>=0)
    #                 {
    #                     vnMatches12[vnMatches21[bestIdx2]]=-1;
    #                     nmatches--;
    #                 }
    #                 vnMatches12[i1]=bestIdx2;
    #                 vnMatches21[bestIdx2]=i1;
    #                 vMatchedDistance[bestIdx2]=bestDist;
    #                 nmatches++;

    #                 if(mbCheckOrientation)
    #                 {
    #                     float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
    #                     if(rot<0.0)
    #                         rot+=360.0f;
    #                     int bin = round(rot*factor);
    #                     if(bin==HISTO_LENGTH)
    #                         bin=0;
    #                     assert(bin>=0 && bin<HISTO_LENGTH);
    #                     rotHist[bin].push_back(i1);
    #                 }
    #             }
    #         }

    #     }

    #     if(mbCheckOrientation)
    #     {
    #         int ind1=-1;
    #         int ind2=-1;
    #         int ind3=-1;

    #         ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

    #         for(int i=0; i<HISTO_LENGTH; i++)
    #         {
    #             if(i==ind1 || i==ind2 || i==ind3)
    #                 continue;
    #             for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
    #             {
    #                 int idx1 = rotHist[i][j];
    #                 if(vnMatches12[idx1]>=0)
    #                 {
    #                     vnMatches12[idx1]=-1;
    #                     nmatches--;
    #                 }
    #             }
    #         }

    #     }

    #     //Update prev matched
    #     for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
    #         if(vnMatches12[i1]>=0)
    #             vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    #     return nmatches;
    # }
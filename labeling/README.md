Hatching your training data: How to label events in R
================
Matthew McKown & Abram Fleishman, Conservation Metrics, Inc
March 2, 2020

<p>

<a href="https://commons.wikimedia.org/wiki/File:Araripe_Manakin_(Antilophia_bokermanni)_on_nest.jpg#/media/File:Araripe_Manakin_(Antilophia_bokermanni)_on_nest.jpg">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Araripe_Manakin_%28Antilophia_bokermanni%29_on_nest.jpg/1200px-Araripe_Manakin_%28Antilophia_bokermanni%29_on_nest.jpg" alt="Araripe Manakin (Antilophia bokermanni) on nest"></a>
<br>By
<a rel="nofollow" class="external text" href="https://www.flickr.com/photos/grandma-shirley/">Hesperia2007
(Shirley)</a>
<a rel="nofollow" class="external text" href="https://www.flickr.com/photos/grandma-shirley/2394332726/">
Araripe Manakin</a>,
<a href="https://creativecommons.org/licenses/by-sa/2.0" title="Creative Commons Attribution-Share Alike 2.0">CC
BY-SA 2.0</a>,
<a href="https://commons.wikimedia.org/w/index.php?curid=5745081">Link</a>

</p>

# Introduction

One of the biggest challenges facing researchers interested in applying
passive acoustic surveys and machine learning techniques to monitor rare
species is the lack of available classification models and/or training
data. Our main article focuses on how one can train classifiers for new
species (signal) using a small training data ste, transfer learning, and
our Biophony model (see \[LINK\]).

In this document, we give an example of how to generate an annotated
training data-set for other species that can be used with teh Biophony
model. The Biophony model makes predictions on discrete, non-overlapping
time-steps (2-second long time steps in this version of the model).
Thus, training data for new classifiers, you will need to include
examples of 2-second sound clips that contain the target signals, and
2-second sound clips that do not contain your target signal (only
ambient sounds).

An easy way to create an approapriaet training dataset is to note the
start and stop time of all of the target signals in each field
recording. Our transfer learning approach can then split each field
recording into a series of consecutive 2-second time steps, and use the
annotated start and stop times to determine which time steps contain the
target signal and which don’t.

Several tools are available to efficiently annotate the start and stop
times of target signals in audio files. Two software tools we often use
in our own work-flows are:

  - [Audacity](https://www.audacityteam.org/) More information on how to
    use that tool to explore and label your recordings can be found
    [here](https://manual.audacityteam.org/man/creating_and_selecting_labels.html).
  - [Raven](https://ravensoundsoftware.com/), is another great tool that
    was specifically developed for bioacoustic analysis by the Cornell
    Laboratory of Ornithology. It is a great option and now has an R
    package to interact with it (
    [Rraven](https://cran.r-project.org/web/packages/Rraven/index.html)
    developed by Marcelo Araya-Salas who also developed the warbleR
    package, *see below*\]).

In this document, we present a workflow that leverages two bioacoustics
analysis packages in [R](https://cran.r-project.org/), a scrip-based
language used by many ecologists/biologists. We also leverage an amazing
open source repository of bird songs from around the world
[**Xenocanto**](https://www.xeno-canto.org/).

To test a real-world problem, we searched the Xenocanto database for
recordings of IUCN Red List species (i.e. species of conservation
concern). Most of the endangered species have few recordings available
(because they are rare, and often in remote areas of the world). We
quickly were drawn to the **Araripe Manakin**
(*Antilophia\_bokermanni*), a stunning species listed as [**Critically
Endangered** by the
IUCN](https://www.iucnredlist.org/species/22728410/130774493). The
species has a very restricted range in Brazil, and is threatened by loss
of habitat among other factors. The species has an active conservation
efforts underway that you can learn about and
[*help*](https://abcbirds.org/bird/araripe-manakin/), and passive
acoustic monitoring coule be a powerful tool for monitoring the species.

At the time we wrote this (Spring 2020), a total of 56 recordings were
available on Xenocanto, with only a few calls per recording. While this
is more data than was available for many IUCN Red List species, it is a
small data-set to train Deep Learning classification models using a
traditional approach. It is thus a great real-world test of the ability
to quickly develop new classifiers with the Biophony model.

Here we walk through the steps we used to create our annotated training
data-set for the Araripe Manakin example in R. Our hope is that this
example will help others generate labeled datasets for other rare
species, and help jump-start the development of new classifiers for
conservation projects. Although we admit the workflow is a little clunky
at times, we think it could be an efficient way for less technical
researchers to generate an appropriate labeled datasets for use with the
Biophony model. We thank the developers of the R libraries we used for
making this workflow possible, and welcome contributions from others
that might help make the annotation process even easier for conservation
biologists.

# 1\. Set things up in R

### Get the packages you need

If you have already installed the needed packages skip this step, if
not, install needed packages here

``` r
# install.packages(c('readr','dplyr','tidyr','fftw','tuneR','seewave','warbleR','monitoR','Cairo','here','purrr'),repos='https://ftp.osuosl.org/pub/cran/')
```

### …and load them.

``` r
library(warbleR)
library(monitoR)
library(tuneR)
library(readr)
library(dplyr)
library(purrr)
library(tidyr)
library(Cairo)
library(here)
```

### Prep the workspace

``` r
# Set paths
download_dir<-here("labeling","ANTBOK")
training_data_dir<-here("labeling","ANTBOK", "training_data")

# Create directories for download
if(!dir.exists(download_dir)){
  dir.create(download_dir) # if the dir does not exist
}
if(!dir.exists(training_data_dir)){
  dir.create(training_data_dir) # if the dir does not exist
}
```

# 2\. Use *warbleR* to download files from Xenocanto

This will download the files and populate a `data.frame` with the
metadata about those files.

``` r
recs<-querxc('Antilophia bokermanni',
             download = F, ## False if re-running and have already downloaded the mp3s 
             X = NULL,
             file.name = c("Genus", "Specific_epithet"),
             parallel = 1, # number of cores to use
             path = download_dir,
             pb = TRUE)
```

    ## 56 recordings found!

``` r
head(recs)
```

    ##   Recording_ID      Genus Specific_epithet Subspecies    English_name
    ## 1       451991 Antilophia       bokermanni            Araripe Manakin
    ## 2       451990 Antilophia       bokermanni            Araripe Manakin
    ## 3       427075 Antilophia       bokermanni            Araripe Manakin
    ## 4       427074 Antilophia       bokermanni            Araripe Manakin
    ## 5       415017 Antilophia       bokermanni            Araripe Manakin
    ## 6       415016 Antilophia       bokermanni            Araripe Manakin
    ##                    Recordist Country                            Locality
    ## 1 JAYRSON ARAUJO DE OLIVEIRA  Brazil    Arajara Park, Barbalha-CE, Ceará
    ## 2 JAYRSON ARAUJO DE OLIVEIRA  Brazil    Arajara Park, Barbalha-CE, Ceará
    ## 3                 Caio Brito  Brazil RPPN Oásis Araripe, Crato-CE, Ceará
    ## 4                 Caio Brito  Brazil    Arajara Park, Barbalha-CE, Ceará
    ## 5              Ross Gallardy  Brazil          Barbalha, CE. Arajara Park
    ## 6              Ross Gallardy  Brazil          Barbalha, CE. Arajara Park
    ##    Latitude Longitude Vocalization_type                           Audio_file
    ## 1 -7.332600 -39.41160              song //www.xeno-canto.org/451991/download
    ## 2 -7.332600 -39.41160              song //www.xeno-canto.org/451990/download
    ## 3 -7.230695 -39.47366             canto //www.xeno-canto.org/427075/download
    ## 4 -7.332630 -39.41163   canto e chamado //www.xeno-canto.org/427074/download
    ## 5 -7.333400 -39.41670              song //www.xeno-canto.org/415017/download
    ## 6 -7.333400 -39.41670              song //www.xeno-canto.org/415016/download
    ##                                        License                         Url
    ## 1 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/451991
    ## 2 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/451990
    ## 3 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/427075
    ## 4 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/427074
    ## 5 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/415017
    ## 6 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/415016
    ##   Quality  Time       Date Altitude
    ## 1       A 10:22 2019-01-07      750
    ## 2       A  9:55 2019-01-07      750
    ## 3       A 09:40 2017-06-04      740
    ## 4       A 11:40 2017-06-05      750
    ## 5       A 10:30 2017-03-04      950
    ## 6       A 10:30 2017-03-04      950
    ##                                                                                              file.name
    ## 1 XC451991-Soldadinho do araripe (Antilophia bokermanni-OK) 7-1-<U+200E>2019 <U+200F><U+200E>10e22.mp3
    ## 2  XC451990-Soldadinho do araripe (Antilophia bokermanni-OK) 7-1-<U+200E>2019 <U+200F><U+200E>9e55.mp3
    ## 3                                  XC427075-Antilophia bokermanni -Caio Brito-(Crato-CE)-BRITO2512.mp3
    ## 4                               XC427074-Antilophia bokermanni -Caio Brito-(Barbalha-CE)-BRITO2525.mp3
    ## 5                                                      XC415017-AraripeManakin_Brazil_030417_song2.mp3
    ## 6                                                       XC415016-AraripeManakin_Brazil_030417_song.mp3
    ##                                                         Spectrogram_small
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-small.png
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-small.png
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-small.png
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-small.png
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-small.png
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-small.png
    ##                                                         Spectrogram_med
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-med.png
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-med.png
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-med.png
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-med.png
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-med.png
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-med.png
    ##                                                         Spectrogram_large
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-large.png
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-large.png
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-large.png
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-large.png
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-large.png
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-large.png
    ##                                                         Spectrogram_full Length
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-full.png   0:20
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-full.png   0:11
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-full.png   0:14
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-full.png   0:53
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-full.png   0:21
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-full.png   0:02
    ##     Uploaded       Other_species
    ## 1 2019-01-17   Turdus leucomelas
    ## 2 2019-01-17 Euphonia chlorotica
    ## 3 2018-07-23                    
    ## 4 2018-07-23                    
    ## 5 2018-05-12                    
    ## 6 2018-05-12                    
    ##                                                                                                                                       Remarks
    ## 1                                                                                                                                            
    ## 2                                                                                                                                            
    ## 3    remarks:Macho adulto;\nhabitat:Mata Ciliar;\nrecorder:Sony PCM-M10;\nmicrophone:Sennheiser ME-066;\nvolume setting:5.;\nweather:Nublado;
    ## 4 remarks:Macho adulto;\nhabitat:Mata Ciliar;\nrecorder:Sony PCM-M10;\nmicrophone:Sennheiser ME-066;\nvolume setting:4.;\nweather:Ensolarado;
    ## 5                                                                                                                                            
    ## 6                                                                                                                                            
    ##   Bird_seen Playback_used Other_species1 Other_species2 Other_species3
    ## 1       yes            no           <NA>           <NA>           <NA>
    ## 2       yes            no           <NA>           <NA>           <NA>
    ## 3   unknown       unknown           <NA>           <NA>           <NA>
    ## 4   unknown       unknown           <NA>           <NA>           <NA>
    ## 5       yes            no           <NA>           <NA>           <NA>
    ## 6       yes            no           <NA>           <NA>           <NA>

### Make a list of the your files and metadata.

We are going to rename the sound files by removing the “-” and replacing
with "\_". We need to add the local path to the file to `recs` so we
will list the files, parse the `filenames`, and join with the `recs` df.

``` r
# list files
file_list<-data.frame(path = list.files(path = download_dir,
                                        pattern = "mp3$",
                                        full.names = T),
                      stringsAsFactors = F) %>%
  mutate(filename = basename(path),
         split_name = gsub(".mp3", "", filename),
         filename_new=gsub("-","_",filename),
         path_new=file.path(dirname(path),filename_new))

file.rename(file_list$path,file_list$path_new)
```

    ##  [1] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
    ## [16] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
    ## [31] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE
    ## [46] TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE

``` r
# Parse name
file_list<-separate(file_list,
                    col = split_name,
                    into = c("Genus", "Specific_epithet","Recording_ID"),
                    sep = "_",
                    remove = T)

# join with the metadata from xenocanto
recs<-recs %>%
  left_join(file_list, by = c("Recording_ID", "Genus", "Specific_epithet"))

head(recs)
```

    ##   Recording_ID      Genus Specific_epithet Subspecies    English_name
    ## 1       451991 Antilophia       bokermanni            Araripe Manakin
    ## 2       451990 Antilophia       bokermanni            Araripe Manakin
    ## 3       427075 Antilophia       bokermanni            Araripe Manakin
    ## 4       427074 Antilophia       bokermanni            Araripe Manakin
    ## 5       415017 Antilophia       bokermanni            Araripe Manakin
    ## 6       415016 Antilophia       bokermanni            Araripe Manakin
    ##                    Recordist Country                            Locality
    ## 1 JAYRSON ARAUJO DE OLIVEIRA  Brazil    Arajara Park, Barbalha-CE, Ceará
    ## 2 JAYRSON ARAUJO DE OLIVEIRA  Brazil    Arajara Park, Barbalha-CE, Ceará
    ## 3                 Caio Brito  Brazil RPPN Oásis Araripe, Crato-CE, Ceará
    ## 4                 Caio Brito  Brazil    Arajara Park, Barbalha-CE, Ceará
    ## 5              Ross Gallardy  Brazil          Barbalha, CE. Arajara Park
    ## 6              Ross Gallardy  Brazil          Barbalha, CE. Arajara Park
    ##    Latitude Longitude Vocalization_type                           Audio_file
    ## 1 -7.332600 -39.41160              song //www.xeno-canto.org/451991/download
    ## 2 -7.332600 -39.41160              song //www.xeno-canto.org/451990/download
    ## 3 -7.230695 -39.47366             canto //www.xeno-canto.org/427075/download
    ## 4 -7.332630 -39.41163   canto e chamado //www.xeno-canto.org/427074/download
    ## 5 -7.333400 -39.41670              song //www.xeno-canto.org/415017/download
    ## 6 -7.333400 -39.41670              song //www.xeno-canto.org/415016/download
    ##                                        License                         Url
    ## 1 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/451991
    ## 2 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/451990
    ## 3 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/427075
    ## 4 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/427074
    ## 5 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/415017
    ## 6 //creativecommons.org/licenses/by-nc-sa/4.0/ //www.xeno-canto.org/415016
    ##   Quality  Time       Date Altitude
    ## 1       A 10:22 2019-01-07      750
    ## 2       A  9:55 2019-01-07      750
    ## 3       A 09:40 2017-06-04      740
    ## 4       A 11:40 2017-06-05      750
    ## 5       A 10:30 2017-03-04      950
    ## 6       A 10:30 2017-03-04      950
    ##                                                                                              file.name
    ## 1 XC451991-Soldadinho do araripe (Antilophia bokermanni-OK) 7-1-<U+200E>2019 <U+200F><U+200E>10e22.mp3
    ## 2  XC451990-Soldadinho do araripe (Antilophia bokermanni-OK) 7-1-<U+200E>2019 <U+200F><U+200E>9e55.mp3
    ## 3                                  XC427075-Antilophia bokermanni -Caio Brito-(Crato-CE)-BRITO2512.mp3
    ## 4                               XC427074-Antilophia bokermanni -Caio Brito-(Barbalha-CE)-BRITO2525.mp3
    ## 5                                                      XC415017-AraripeManakin_Brazil_030417_song2.mp3
    ## 6                                                       XC415016-AraripeManakin_Brazil_030417_song.mp3
    ##                                                         Spectrogram_small
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-small.png
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-small.png
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-small.png
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-small.png
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-small.png
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-small.png
    ##                                                         Spectrogram_med
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-med.png
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-med.png
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-med.png
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-med.png
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-med.png
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-med.png
    ##                                                         Spectrogram_large
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-large.png
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-large.png
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-large.png
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-large.png
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-large.png
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-large.png
    ##                                                         Spectrogram_full Length
    ## 1 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451991-full.png   0:20
    ## 2 //www.xeno-canto.org/sounds/uploaded/LXKLWEDKEM/ffts/XC451990-full.png   0:11
    ## 3 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427075-full.png   0:14
    ## 4 //www.xeno-canto.org/sounds/uploaded/YWWWUBVAJF/ffts/XC427074-full.png   0:53
    ## 5 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415017-full.png   0:21
    ## 6 //www.xeno-canto.org/sounds/uploaded/FNIOJOZADD/ffts/XC415016-full.png   0:02
    ##     Uploaded       Other_species
    ## 1 2019-01-17   Turdus leucomelas
    ## 2 2019-01-17 Euphonia chlorotica
    ## 3 2018-07-23                    
    ## 4 2018-07-23                    
    ## 5 2018-05-12                    
    ## 6 2018-05-12                    
    ##                                                                                                                                       Remarks
    ## 1                                                                                                                                            
    ## 2                                                                                                                                            
    ## 3    remarks:Macho adulto;\nhabitat:Mata Ciliar;\nrecorder:Sony PCM-M10;\nmicrophone:Sennheiser ME-066;\nvolume setting:5.;\nweather:Nublado;
    ## 4 remarks:Macho adulto;\nhabitat:Mata Ciliar;\nrecorder:Sony PCM-M10;\nmicrophone:Sennheiser ME-066;\nvolume setting:4.;\nweather:Ensolarado;
    ## 5                                                                                                                                            
    ## 6                                                                                                                                            
    ##   Bird_seen Playback_used Other_species1 Other_species2 Other_species3
    ## 1       yes            no           <NA>           <NA>           <NA>
    ## 2       yes            no           <NA>           <NA>           <NA>
    ## 3   unknown       unknown           <NA>           <NA>           <NA>
    ## 4   unknown       unknown           <NA>           <NA>           <NA>
    ## 5       yes            no           <NA>           <NA>           <NA>
    ## 6       yes            no           <NA>           <NA>           <NA>
    ##                                                                          path
    ## 1 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_451991.mp3
    ## 2 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_451990.mp3
    ## 3 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_427075.mp3
    ## 4 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_427074.mp3
    ## 5 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_415017.mp3
    ## 6 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_415016.mp3
    ##                           filename                     filename_new
    ## 1 Antilophia_bokermanni_451991.mp3 Antilophia_bokermanni_451991.mp3
    ## 2 Antilophia_bokermanni_451990.mp3 Antilophia_bokermanni_451990.mp3
    ## 3 Antilophia_bokermanni_427075.mp3 Antilophia_bokermanni_427075.mp3
    ## 4 Antilophia_bokermanni_427074.mp3 Antilophia_bokermanni_427074.mp3
    ## 5 Antilophia_bokermanni_415017.mp3 Antilophia_bokermanni_415017.mp3
    ## 6 Antilophia_bokermanni_415016.mp3 Antilophia_bokermanni_415016.mp3
    ##                                                                      path_new
    ## 1 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_451991.mp3
    ## 2 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_451990.mp3
    ## 3 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_427075.mp3
    ## 4 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_427074.mp3
    ## 5 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_415017.mp3
    ## 6 D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_415016.mp3

# 3\. Intro to using *monitoR* to annotate target signals

We used the `monitoR` package to interactively draw boxes around target
signals in spectrograms of each recording. The monitoR package includes
the `viewSpec` function which you can use to interactively annotate the
location of target signals on spectrograms, and label these with a
species tag (We use the official 6-letter code for the Araripe Manakin -
*ANTBOK*). We modified the `monitoR::viewSpec` to allow us to save the
output with the same file name as the input sound, allowing the workflow
to move quicker.

``` r
source(here("labeling","viewSpec_cmi.R"))
```

In this example we review and label recordings in 2-second increments,
because that is the time-step used by the ***Biophony model*** for
training and prediction.

The `viewSpec` labeling process is interactive. When you run the script
R will open spectrograms of each 2-second clip in an external graphics
window. It is hard to capture the labeling process in an R markdown
notebook, so here are some brief pointers and screen shots to help guide
you.

<img src="https://github.com/microsoft/acoustic-bird-detection/blob/master/labeling/Images/Start_labeling_viewSpec_1.png" width="100%" style="display: block; margin: auto;" />

### Walk-through of the iterative process:

1.  Change the index value to pick a recording from the list of files
    you downloaded.
2.  Open a sound file.
3.  Run the `viewSpec_cmi` function script:

<!-- end list -->

  - Use the menu choices in the R Console to navigate the sound file
  - `return` will move you forward, and `b` + `return` will move back
  - `p` followed by `return` to play the clip
  - `a` will start an annotation
  - Click the crosshair on the upper left corner of a box around your
    signal
  - Click on the lower right corner of a box to enclose your signal
  - Enter a labeled name in the Console (‘ANTBOK’ in this case)  
  - Then ‘right-click’ twice in the spectrogram to stop the annotation
    process (the clunkiest part of this process).
  - Move on to review and label all of the target sounds in the file
    \[NOTE - the training process assumes that all target sounds have
    been labeled, and all other sections of the recording are
    “non-target” signals\].
  - Quit, and a `csv` with the data for your labeled data boxes will be
    saved using the name of the sound file you are using.

<!-- end list -->

4.  Go to Step 1

<img src="https://github.com/microsoft/acoustic-bird-detection/blob/master/labeling/Images/song_labeling_all.png" style="display: block; margin: auto;" />

You will need to follow this process for each sound file. Our script
allows you to change an index value to move between the list of
downloaded files in order, and you can take a break between files and
pick back up where you left off. The workflow takes some getting used
to, and involves a lot of clicking between windows. But we got pretty
efficient at it after a few sessions.

# 4\. Start labeling

\[NOTE - this section will walk you through the interactive process of
annotating sound files. If you are not interested in actively labeling
at this time, you can skip to Step XXXXXX to see the format of the
completed training dataset\]

### Pick a file.

Pick the file you want to work on. First set an index value to view and
label a file on your list (i.e. i=1, i=2, i=3…i=56)

``` r
i=1
```

### Open the audio file.

``` r
csv_filename<-gsub("-",'_',recs$filename_new[i])
File2label<-readMP3(file.path(recs$path_new[i]))

if(Sys.info()['sysname']!="Windows"){
  CairoX11(width = 5, height = 5) # this would open it in a new window
}else{
  x11(width = 5, height = 5)
}
This_file_name<-recs$filename[i]
Number_of_files <-length(recs$filename)
```

### Label sounds in spectrograms.

**The current file is:** *Antilophia\_bokermanni\_451991.mp3* (**1** of
**56**)

``` r
viewSpec_cmi(clip = File2label,
             interactive =TRUE,
             start.time = 0,
             units = "seconds",
             page.length = 2,
             annotate = TRUE,
             channel = "left",
             output.dir = download_dir, # this is the same as the sounds but you could change it
             file.name=gsub("mp3",'csv',csv_filename), # this will name the same as the sound file
             frq.lim = c(0, 12),
             spec.col = gray.3(),
             page.ovlp = 0,
             player = "afplay",
             wl = 512, ovlp = 0,
             wn = "hanning",
             consistent = TRUE,
             mp3.meta = list(kbps = 128, samp.rate = 44100, stereo = TRUE),
             main = csv_filename)
```

### Repeat until you have reviewed and labeled all files.

# 5\. Concatenate all `csv` files into a training dataset

``` r
myfiles = list.files(path=download_dir, pattern="*.csv", full.names=TRUE)

myfiles[1]
```

    ## [1] "D:/CM,Inc/git_repos/4Earth/labeling/ANTBOK/Antilophia_bokermanni_118707.csv"

``` r
# read in each label csv and bind them into a long data.frame
all_labels <-  purrr::map_dfr(myfiles, read_csv,
                              col_types=cols(
                                file_name = col_character(),
                                start.time = col_double(),
                                end.time = col_double(),
                                min.frq = col_double(),
                                max.frq = col_double(),
                                label = col_character(),
                                name = col_character()))%>% 
  mutate(filename = gsub("-", "_",file_name)) %>% 
  select (-c(name, file_name)) %>% 
  select(filename, everything()) %>% 
  mutate(box_length=end.time-start.time)

# join with meta data
training_labels<- all_labels %>% 
  left_join(recs, by='filename')


write_csv(training_labels,
          path = file.path(training_data_dir,"ANTBOK_training_labels.csv"))
```

[Here](https://github.com/microsoft/acoustic-bird-detection/blob/master/labeling/ANTBOK/training_data/ANTBOK_training_labels.csv) is a link to the label file that we use for the next tutorial so you can see the correct format for ingusting into the next step.

You are now ready to train your new classifier. Follow [this
link](https://github.com/microsoft/acoustic-bird-detection)
to get started on the next step.

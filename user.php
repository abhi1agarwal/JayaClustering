<?php

    include("./includes/preProcess.php");
    
?>

<!doctype html>
<html lang="en">
<head>
	<meta charset="utf-8" />
	<link rel="apple-touch-icon" sizes="76x76" href="assets/img/apple-icon.png">
	<link rel="icon" type="image/png" sizes="96x96" href="assets/img/favicon.png">
	<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />

	<title>MNNIT - DDPC</title>

	<meta content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0' name='viewport' />
    <meta name="viewport" content="width=device-width" />


    <!-- Bootstrap core CSS     -->
    <link href="assets/css/bootstrap.min.css" rel="stylesheet" />

    <!-- Animation library for notifications   -->
    <link href="assets/css/animate.min.css" rel="stylesheet"/>

    <!--  Paper Dashboard core CSS    -->
    <link href="assets/css/paper-dashboard.css" rel="stylesheet"/>

    <!--  CSS for Demo Purpose, don't include it in your project     -->
    <link href="assets/css/demo.css" rel="stylesheet" />

    <!--  Fonts and icons     -->
    <link href="http://maxcdn.bootstrapcdn.com/font-awesome/latest/css/font-awesome.min.css" rel="stylesheet">
    <link href='https://fonts.googleapis.com/css?family=Muli:400,300' rel='stylesheet' type='text/css'>
    <link href="assets/css/themify-icons.css" rel="stylesheet">

    <link href="./css/myCss.css" rel="stylesheet">

</head>
<body>

<div class="wrapper">
	<div class="sidebar" data-background-color="white" data-active-color="danger">

    <!--
		Tip 1: you can change the color of the sidebar's background using: data-background-color="white | black"
		Tip 2: you can change the color of the active button using the data-active-color="primary | info | success | warning | danger"
	-->

    	<div class="sidebar-wrapper">
            <div class="logo">
                <?php include('./includes/topleft.php') ?>
            </div>

            <?php

                $currentTab = "user";

                include("./includes/sideNav.php");

            ?>

    	</div>
    </div>

    <div class="main-panel">
		<nav class="navbar navbar-default">
            <div class="container-fluid">
                <div class="navbar-header">
                    <?php include('./includes/logo.php'); ?>
                </div>
                <div class="collapse navbar-collapse">
                    <ul class="nav navbar-nav navbar-right">
                        <li>
                            <a href="#" class="dropdown-toggle" data-toggle="dropdown">
                                <i class="ti-panel"></i>
								<p>Stats</p>
                            </a>
                        </li>
                        <?php include('./includes/notifications.php'); ?>
						<li>
                            <a href="./logout.php">
                                <i class="ti-settings"></i>
                                <p>LogOut</p>
                            </a>
                        </li>
                    </ul>

                </div>
            </div>
        </nav>


        <div class="content">
            <div class="container-fluid">
                <div class="row">
                    <div class="col-lg-4 col-md-5">
                        <div class="card card-user">
                            <div class="image">
                                <img src="assets/img/background.jpg" alt="..."/>
                            </div>
                            <div class="content">
                                <div class="author">
                                  <img class="avatar border-white" src="<?php echo $user['photo_path']; ?>" alt="..."/>
                                  
                                  <form method="post" action="updatePic.php" enctype="multipart/form-data">
                                        <input type="file" name="photo" id="photo">
                                        <input type="submit" name="submit" value="Upload Image">
                                  </form>
                                  <h4 class="title"><?php echo $name; ?><br />
                                     <!-- <a href="#"><small>@chetfaker</small></a> -->
                                  </h4>
                                </div>
                                <!-- <p class="description text-center">
                                    "I like the way you work it <br>
                                    No diggity <br>
                                    I wanna bag it up"
                                </p> -->
                            </div>
                            <hr>
                            <!-- <div class="text-center">
                                <div class="row">
                                    <div class="col-md-3 col-md-offset-1">
                                        <h5>12<br /><small>Files</small></h5>
                                    </div>
                                    <div class="col-md-4">
                                        <h5>2GB<br /><small>Used</small></h5>
                                    </div>
                                    <div class="col-md-3">
                                        <h5>24,6$<br /><small>Spent</small></h5>
                                    </div>
                                </div>
                            </div> -->
                        </div>
                       <!-- <div class="card">
                            <div class="header">
                                <h4 class="title">Upload Documents</h4>
                            </div>
                            <div class="content">
                                <form action="uploadDoc.php" enctype="multipart/form-data" method="post">
                                Doc Type : <select name="application_type">
                                    <?php
                                        $query = "SELECT * FROM documentlookup";
                                        $documentTypes = mysqli_query($connection, $query);
                                        
                                        while( $thisDocumentType = mysqli_fetch_array($documentTypes)  )
                                        {
                                    ?>
                                        <option value="<?php echo $thisDocumentType['doc_type_id'] ?>"><?php echo $thisDocumentType['doc_type'] ?></option>
                                    <?php
                                        }
                                    ?>
                                </select>

                                <input type="file" name="doc" value="" /><br />
                                <input type="submit" name="submit" value="Upload Doc" />
                                </form>
                            </div>
                        </div>-->
                        <!-- <div class="card">
                            <div class="header">
                                <h4 class="title">Team Members</h4>
                            </div>
                            <div class="content">
                                <ul class="list-unstyled team-members">
                                            <li>
                                                <div class="row">
                                                    <div class="col-xs-3">
                                                        <div class="avatar">
                                                            <img src="assets/img/faces/face-0.jpg" alt="Circle Image" class="img-circle img-no-padding img-responsive">
                                                        </div>
                                                    </div>
                                                    <div class="col-xs-6">
                                                        DJ Khaled
                                                        <br />
                                                        <span class="text-muted"><small>Offline</small></span>
                                                    </div>

                                                    <div class="col-xs-3 text-right">
                                                        <btn class="btn btn-sm btn-success btn-icon"><i class="fa fa-envelope"></i></btn>
                                                    </div>
                                                </div>
                                            </li>
                                            <li>
                                                <div class="row">
                                                    <div class="col-xs-3">
                                                        <div class="avatar">
                                                            <img src="assets/img/faces/face-1.jpg" alt="Circle Image" class="img-circle img-no-padding img-responsive">
                                                        </div>
                                                    </div>
                                                    <div class="col-xs-6">
                                                        Creative Tim
                                                        <br />
                                                        <span class="text-success"><small>Available</small></span>
                                                    </div>

                                                    <div class="col-xs-3 text-right">
                                                        <btn class="btn btn-sm btn-success btn-icon"><i class="fa fa-envelope"></i></btn>
                                                    </div>
                                                </div>
                                            </li>
                                            <li>
                                                <div class="row">
                                                    <div class="col-xs-3">
                                                        <div class="avatar">
                                                            <img src="assets/img/faces/face-3.jpg" alt="Circle Image" class="img-circle img-no-padding img-responsive">
                                                        </div>
                                                    </div>
                                                    <div class="col-xs-6">
                                                        Flume
                                                        <br />
                                                        <span class="text-danger"><small>Busy</small></span>
                                                    </div>

                                                    <div class="col-xs-3 text-right">
                                                        <btn class="btn btn-sm btn-success btn-icon"><i class="fa fa-envelope"></i></btn>
                                                    </div>
                                                </div>
                                            </li>
                                        </ul>
                            </div>
                        </div> -->
                    </div>
                    <div class="col-lg-8 col-md-7">
                        <div class="card">
                            <div class="header">
                                <h4 class="title">Edit Profile</h4>
                            </div>
                            <div class="content">
                                <form method="GET" action="updateProfile.php">
                                    <div class="row">
                                        <div class="col-md-3">
                                            <div class="form-group">
                                                <label>Registration Number</label>
                                                <input type="text" class="form-control border-input" disabled placeholder="Company" value="<?php echo $user['reg_no']; ?>">
                                            </div>
                                        </div>
                                        <div class="col-md-9">
                                            <div class="form-group">
                                                <label>Name</label>
                                                <input type="text" class="form-control border-input" placeholder="name" value="<?php echo $user['name'] ?>" name="name">
                                            </div>
                                        </div>
                                        
                                    </div>

                                    <div class="row">
                                        <div class="col-md-6">
                                            <div class="form-group">
                                                <label for="exampleInputEmail1">Email address</label>
                                                <input type="email" class="form-control border-input" placeholder="mail_id" value="<?php echo $user['mail_id'] ?>" name="mail_id">
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <div class="form-group">
                                                <label for="exampleInputEmail1">Contact Number</label>
                                                <input type="text" class="form-control border-input" placeholder="contact_no" value="<?php echo $user['contact_no'] ?>" name="contact_no">
                                            </div>
                                        </div>
                                    </div>

                                    <div class="row">
                                        <div class="col-md-12">
                                            <div class="form-group">
                                                <label>Address</label>
                                                <input type="text" class="form-control border-input" placeholder="Home Address" value="<?php echo $user['address'] ?>" name="address">
                                            </div>
                                        </div>
                                    </div>

                                    <div class="row">
                                        <div class="col-md-12">
                                            <div class="form-group">
                                                <label>Father's Name</label>
                                                <input type="text" class="form-control border-input" placeholder="Father's Name" value="<?php echo $user['father_name'] ?>" name = "father_name">
                                            </div>
                                        </div>
                                    </div>

                                    <!-- <div class="row">
                                        <div class="col-md-12">
                                            <div class="form-group">
                                                <label>About Me</label>
                                                <textarea rows="5" class="form-control border-input" placeholder="Here can be your description" value="Mike">Oh so, your weak rhyme
You doubt I'll bother, reading into it
I'll probably won't, left to my own devices
But that's the difference in our opinions.</textarea>
                                            </div>
                                        </div>
                                    </div> -->
                                    <div class="text-center">
                                        <button type="submit" class="btn btn-info btn-fill btn-wd">Update Profile</button>
                                    </div>
                                    <div class="clearfix"></div>
                                </form>
                            </div>
                        </div>
                    </div>


                </div>
            </div>
        </div>


        <footer class="footer">
            <!-- <div class="container-fluid">
                <nav class="pull-left">
                    <ul>

                        <li>
                            <a href="http://www.creative-tim.com">
                                Creative Tim
                            </a>
                        </li>
                        <li>
                            <a href="http://blog.creative-tim.com">
                               Blog
                            </a>
                        </li>
                        <li>
                            <a href="http://www.creative-tim.com/license">
                                Licenses
                            </a>
                        </li>
                    </ul>
                </nav>
				<div class="copyright pull-right">
                    &copy; <script>document.write(new Date().getFullYear())</script>, made with <i class="fa fa-heart heart"></i> by <a href="http://www.creative-tim.com">Creative Tim</a>
                </div>
            </div> -->
        </footer>

    </div>
</div>


</body>

    <!--   Core JS Files   -->
    <script src="assets/js/jquery-1.10.2.js" type="text/javascript"></script>
	<script src="assets/js/bootstrap.min.js" type="text/javascript"></script>

	<!--  Checkbox, Radio & Switch Plugins -->
	<script src="assets/js/bootstrap-checkbox-radio.js"></script>

	<!--  Charts Plugin -->
	<script src="assets/js/chartist.min.js"></script>

    <!--  Notifications Plugin    -->
    <script src="assets/js/bootstrap-notify.js"></script>

    <!--  Google Maps Plugin    -->
    <script type="text/javascript" src="https://maps.googleapis.com/maps/api/js"></script>

    <!-- Paper Dashboard Core javascript and methods for Demo purpose -->
	<script src="assets/js/paper-dashboard.js"></script>

	<!-- Paper Dashboard DEMO methods, don't include it in your project! -->
	<script src="assets/js/demo.js"></script>

    <script type="text/javascript">

        function removeNot() {

            $('.notificationAlert').css({
                'display': 'none'
            });

            xmldata = new XMLHttpRequest();

            var el = document.getElementById('notid').innerHTML;

            var urltosend = "set_cookie.php?notid="+el;
            console.log(el);
            xmldata.open("GET", urltosend,false);
            xmldata.send(null);
            if(xmldata.responseText != ""){
                toPrint = xmldata.responseText;
            }

            console.log(toPrint);


            // body...
        }
    </script>

</html>

<?php

session_start();
if(!isset($_SESSION['reg_no']))
{
    header("location: ./");
}
else
    $reg_no = $_SESSION['reg_no'];

include("./includes/connect.php");
 
define ("filesplace","./docs/");

if (is_uploaded_file($_FILES['doc']['tmp_name']))
{

	if ($_FILES['doc']['type'] !== 'application/pdf') 
	{
		header("location: ./uploadDocument.php?doc_type=1");
	} 
	else 
	{
		$query = "SELECT * FROM document";
		$docs = mysqli_query($connection, $query);
		$docCount = mysqli_num_rows($docs);
		$docCount = $docCount + 1;
		$name = (string)$docCount;
		$docType = $_POST['application_type'];
		$currDate = (string)date("Y-m-d");
		$result = move_uploaded_file($_FILES['doc']['tmp_name'], filesplace."/$name.pdf");

		$query = "SELECT sem_no FROM studentregistration WHERE `reg_no` = '$reg_no'";
		$record = mysqli_query($connection, $query);
		$row = mysqli_fetch_array($record);
		$sem_no = $row['sem_no'];

		$year = "2016";

		if ($result == 1)
		{
			$query = "INSERT INTO document (`doc_id`, `member_id`, `sem_no`, `academic_year`, `application_type`, `date_of_upload`) VALUES ('$name', '$reg_no', '$sem_no', '$year', '$docType', '$currDate')";
			//echo $query;
			mysqli_query($connection, $query);
			header('Location: user.php');
		}
		else echo "<p>Sorry, Error happened while uploading . </p>";
	} 
}
else
{
	header("location: ./uploadDocument.php?doc_type=0");
}

?>